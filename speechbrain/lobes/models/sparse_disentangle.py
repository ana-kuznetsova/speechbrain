from math import tau

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import logging


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone() 

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0, max_steps=5000):
        super().__init__()
        self.alpha = alpha
        self.max_steps = max_steps

    def forward(self, x, current_step=None):
        """
        Args:
            x: Input tensor.
            current_step: The global training step passed from your optimizer loop.
        """
        if self.training and current_step is not None:
            # Linear scheduler capped at max alpha
            active_alpha = min(self.alpha, self.alpha * (current_step / self.max_steps))
        elif self.training and current_step is None:
            # Fallback if you forget to pass the step during training
            active_alpha = self.alpha
        else:
            active_alpha = 0.0 # No reversal during validation/testing
            
        return GradientReversalFunction.apply(x, active_alpha)

    

class DecoderAdapter(nn.Module):
    def __init__(self, input_dim=64, output_dim=1024):
        super().__init__()
        # First linear layer to expand capacity
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        
        # Final projection to DAC dimension (Strictly NO activation at the end!)
        self.fc2 = nn.Linear(512, output_dim)
        
        # Normalize the outputs to help match the variance of the ±18 range
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x) 
        return x

class SparseLayerMixer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

    def forward(self, h_stacked, mid, type="cnt"):
        """
        h_stacked shape: [B, num_layers, T, dict_dim]
        """
        # 1. Slice content subspace for each sparse layer
        if type == "cnt":
            h_layers = h_stacked[:, :, :mid, ].permute(0,1,3,2).contiguous()
        elif type == "spk":
            h_layers = h_stacked[:, :, mid:, ].permute(0,1,3,2).contiguous()
        else:
            raise ValueError("type must be 'cnt' or 'spk'")

        # 2. Concatenate layer-wise content features.
        # [B, L, T, C] -> [B, T, L, C] -> [B, T, L*C]
        h_cnt_spk = h_layers.permute(0, 2, 1, 3).contiguous()
        h_cnt_spk = h_cnt_spk.view(h_cnt_spk.shape[0], h_cnt_spk.shape[1], -1)

        return h_cnt_spk

class SparseDisentangle(nn.Module):
    """Sparse dictionary module using unrolled ISTA iterations with asymmetric latent codes:

      - H_cnt: Time-varying content matrix of shape (B, K_cnt, T)
      - H_spk: Time-invariant speaker vector of shape (B, K_spk, 1)

    Reconstruction: z_approx = W_cnt @ H_cnt + W_spk @ H_spk

    Args:
        input_dim (int): Feature dimension D of input signal z.
        dict_dim (int): Total dictionary size K = K_cnt + K_spk.
        content_ratio (float): Fraction of dictionary atoms assigned to content
          W_cnt.
        num_steps (int): Number of unrolled ISTA steps.
        step_size (float): ISTA step size eta.
        l1_lambda (float): Sparsity penalty lambda.
    """

    def __init__(
        self,
        input_dim: int,
        dict_dim: int,
        num_speakers: int = 252,
        content_ratio: float = 0.5,
        num_steps: int = 10,
        step_size: float = 2.0,
        l1_lambda: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.content_ratio = content_ratio
        self.num_steps = num_steps
        self.step_size = step_size
        self.l1_lambda = l1_lambda

        # Split sizes
        self.mid = int(dict_dim * content_ratio)

        # Dictionary parameter W (input_dim x dict_dim)
        self.W = nn.Parameter(torch.randn(input_dim, dict_dim) * 0.1)
        self.prototype_table = nn.Embedding(num_speakers, dict_dim - self.mid)
        self.register_buffer(
            "speaker_initialized",
            torch.zeros(num_speakers, dtype=torch.bool),
        )
        self.speaker_frame_map = {}

    def _get_or_init_speaker_prototypes(self, H_spk_time: torch.Tensor, speaker_codes: torch.Tensor) -> torch.Tensor:
        """Initializes or retrieves prototype vectors for each sample in the batch.

        Args:
            H_spk_time: Projected speaker frame representations (B, K_spk, T)
            speaker_codes: Tensor of speaker IDs (B,)

        Returns:
            H_spk: Initialized/looked-up speaker vectors (B, K_spk, 1)
        """
        B, K_spk, T = H_spk_time.shape
        spk_ids = speaker_codes.to(
            device=H_spk_time.device, dtype=torch.long
        ).view(-1)

        if spk_ids.numel() == 1 and B > 1:
            spk_ids = spk_ids.expand(B)

        batch_prototypes = []

        for i, spk_id_tensor in enumerate(spk_ids):
            spk_id = spk_id_tensor.item()

            # 1. Assign a fixed random frame index for this speaker if unseen
            if spk_id not in self.speaker_frame_map:
                self.speaker_frame_map[spk_id] = torch.randint(
                    0, T, size=(1,)
                ).item()

            spk_idx = self.speaker_frame_map[spk_id] % T

            # 2. If prototype table entry is uninitialized, copy the sampled frame vector
            if not self.speaker_initialized[spk_id].item():
                sampled_vec = H_spk_time[i, :, spk_idx]  # Shape: (K_spk,)
                with torch.no_grad():
                    self.prototype_table.weight[spk_id].copy_(sampled_vec)
                    self.speaker_initialized[spk_id] = True

            # 3. Lookup current embedding from prototype_table
            proto_vec = self.prototype_table(
                spk_ids[i : i + 1]
            )  # Shape: (1, K_spk)

            batch_prototypes.append(proto_vec)

        # Stack into (B, K_spk, 1)
        prototypes = torch.cat(batch_prototypes, dim=0).unsqueeze(-1)
        return prototypes

    def soft_threshold(self, H: torch.Tensor, threshold: float) -> torch.Tensor:
        """Applies proximal operator: sign(H) * max(0, |H| - threshold)."""
        return torch.sign(H) * torch.relu(torch.abs(H) - threshold)

    def forward(
        self, z: torch.Tensor, speaker_codes: torch.Tensor = None
    ) -> tuple:
        if speaker_codes is None:
            raise ValueError(
                "speaker_codes are required to track speaker prototypes"
            )
        B, D_in, T = z.shape

        # 1. Normalize dictionary columns
        W_norm = torch.linalg.vector_norm(
            self.W, ord=2, dim=0, keepdim=True
        ).clamp(min=1e-5)
        W_normalized = self.W / W_norm  # Shape: (D_in, K)

        # Split normalized dictionary into W_cnt and W_spk
        W_cnt = W_normalized[:, : self.mid]  # (D_in, K_cnt)
        W_spk = W_normalized[:, self.mid :]  # (D_in, K_spk)

        # 2. Initializations
        H_cnt = torch.einsum("dk, bdt -> bkt", W_cnt, z)  # (B, K_cnt, T)

        # H_spk prototype lookup: (B, K_spk, 1)
        H_spk_time = torch.einsum("dk, bdt -> bkt", W_spk, z)
        H_spk = self._get_or_init_speaker_prototypes(
            H_spk_time, speaker_codes
        )  # (B, K_spk, 1)

        # 3. Dynamic Lipschitz Step Size Calculation
        with torch.no_grad():
            s = torch.linalg.svdvals(W_normalized)[0]
            L = (s**2).item()
            # Safe step size for joint descent
            eta = 1.0 / max(L, 1e-5)
            threshold = eta * self.l1_lambda

        # 4. ISTA Optimization Loop

        # 4. ISTA Optimization Loop
        for step in range(self.num_steps):
            # z_cnt: (B, D_in, T)
            z_cnt = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt)
            
            # z_spk: (B, D_in, 1) -- contract K_spk, keep single frame dimension 'r' (size 1)
            z_spk = torch.einsum("dk, bkr -> bdr", W_spk, H_spk)
            
            # PyTorch automatically broadcasts z_spk (B, D_in, 1) across T when subtracting
            residual = z - (z_cnt + z_spk)

            # Content update: (B, K_cnt, T)
            grad_cnt = torch.einsum("dk, bdt -> bkt", W_cnt, residual)
            H_cnt = self.soft_threshold(H_cnt + eta * grad_cnt, threshold)

            # Speaker update: contract d and t -> (B, K_spk), unsqueeze to (B, K_spk, 1)
            grad_spk = torch.einsum("dk, bdt -> bk", W_spk, residual).unsqueeze(-1) / T
            H_spk = self.soft_threshold(H_spk + eta * grad_spk, threshold)

            #with torch.no_grad():
            #    residual_norm = residual.norm().item()
            #    print(f"Step {step}: Residual Norm = {residual_norm:.4f}")

        # 5. Final Reconstruction
        z_cnt = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt)
        z_spk = torch.einsum("dk, bkr -> bdr", W_spk, H_spk)
        z_approx = z_cnt + z_spk  # Broadcasts (B, D_in, 1) across T

        # 6. Regularization penalties
        l1_cnt = H_cnt.abs().mean()
        l1_spk = H_spk.abs().mean()

        return z_approx, H_cnt, H_spk, l1_cnt, l1_spk

class ResidualSparseDisentangle(nn.Module):
    def __init__(
        self,
        input_dim,
        dict_dim,
        num_sparse_layers=4,
        content_ratio=0.5,
        num_speakers=252,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.num_layers = num_sparse_layers
        self.content_ratio = content_ratio

        self.sparse_module_list = nn.ModuleList(
            [
                SparseDisentangle(
                    input_dim,
                    dict_dim,
                    content_ratio=content_ratio,
                    num_speakers=num_speakers
                )
                for _ in range(num_sparse_layers)
            ]
        )

        mid = int(dict_dim * content_ratio)
        spk_dim = dict_dim - mid
        self.spk_concat_dim = spk_dim * num_sparse_layers
        self.cnt_concat_dim = mid * num_sparse_layers

        self.layer_mixer = SparseLayerMixer(num_sparse_layers)

        self.decoder_adapter = DecoderAdapter(
            self.cnt_concat_dim + self.spk_concat_dim,
            input_dim,
        )

    def _build_content_features(self, h_stacked, mid):
        return self.layer_mixer(h_stacked, mid, type="cnt")
    
    def forward(self, z, speaker_codes=None, **kwargs):
        residual = z
        all_h = []
        total_reconstruction = 0
        total_l1_reg_content = 0.0
        total_l1_reg_speaker = 0.0

        for sparse_module in self.sparse_module_list:
            x_approx_i, h_cnt_i, h_spk_i, l1_cnt_i, l1_spk_i = sparse_module(residual, speaker_codes=speaker_codes)
            residual = residual - x_approx_i
            total_reconstruction = total_reconstruction + x_approx_i
            # Expand speaker code over time so shape is (B, D, T).
            h_spk_time = h_spk_i.expand(-1, h_spk_i.shape[1], z.shape[2]).contiguous()

            h_i = torch.cat([h_cnt_i, h_spk_time], dim=1)

            # Fix the loss to compute once instead of the iterations
            all_h.append(h_i)
            total_l1_reg_content += l1_cnt_i
            total_l1_reg_speaker += l1_spk_i

        h_stacked = torch.stack(all_h, dim=1)
        mid = int(h_stacked.shape[-2] * self.content_ratio)

        h_cnt = self._build_content_features(h_stacked, mid)
        h_spk = self.layer_mixer(h_stacked, mid, type="spk")

        h_projected = torch.cat([h_cnt, h_spk], dim=-1)
        h_projected = self.decoder_adapter(h_projected)

        adapter_loss = F.mse_loss(
            h_projected.view_as(z),
            z,
        )
        
        total_recon_loss = F.mse_loss(
            total_reconstruction,
            z
        )

        return (
            #z_proj_content,
            #z_proj_speaker,
            h_cnt,
            h_spk,
            h_stacked,
            h_projected,
            total_recon_loss,
            total_l1_reg_content,
            total_l1_reg_speaker,
            adapter_loss,
        )
