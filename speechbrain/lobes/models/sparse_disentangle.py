from math import tau

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import speechbrain as sb
import logging

def l12_cnt_loss(H_cnt: torch.Tensor) -> torch.Tensor:
    """Computes the L12 norm for the content matrix H_cnt."""
    l1_per_time = torch.sum(torch.abs(H_cnt), dim=1)
    l12_per_sample = torch.sqrt(torch.sum(l1_per_time ** 2, dim=1) + 1e-8)
    return l12_per_sample.mean()


def l1_spk_loss(H_spk: torch.Tensor) -> torch.Tensor:
    """Computes the L1 norm for the speaker matrix H_spk."""
    loss = torch.abs(H_spk).sum(dim=-1)
    return loss.mean()

def recon_loss(z: torch.Tensor, z_approx: torch.Tensor) -> torch.Tensor:
    """Computes the reconstruction loss between z and z_approx."""
    loss = F.mse_loss(z_approx, z, reduction='none')
    loss = torch.sum(loss, dim=[1, 2])
    return loss.mean()

def cnt_mask(z: torch.Tensor, wav_lens: torch.Tensor = None) -> tuple:
    """Returns the valid-frame mask (B, 1, T) and valid frame counts (B, 1)."""
    B, _, T = z.shape
    if wav_lens is None:
        mask = torch.ones(B, 1, T, device=z.device, dtype=z.dtype)
        n_valid = torch.full((B, 1), float(T), device=z.device)
        return mask, n_valid
    lens = torch.round(wav_lens.to(z.device) * T).clamp(min=1, max=T)
    mask = (
        (torch.arange(T, device=z.device)[None, :] < lens[:, None])
        .unsqueeze(1)
        .to(z.dtype)
    )
    return mask, mask.sum(dim=-1)

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



class SparseDisentangle(nn.Module):
    """Sparse dictionary module using unrolled ISTA iterations with functional asymmetric latent codes:

      - H_cnt: Time-varying content matrix of shape (B, K_cnt, T)
      - H_spk: Time-invariant speaker vector of shape (B, K_spk, 1) derived functionally from z.

    Reconstruction: z_approx = W_cnt @ H_cnt + W_spk @ H_spk
    """

    def __init__(
        self,
        input_dim: int,
        dict_dim: int,
        content_ratio: float = 0.5,
        num_steps: int = 5,
        l1_lambda: float = 1e-4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.content_ratio = content_ratio
        self.num_steps = num_steps
        self.l1_lambda = l1_lambda
    
        # Split sizes
        self.mid = int(dict_dim * content_ratio)

        # Dictionary parameter W (input_dim x dict_dim)
        self.W = nn.Parameter(torch.randn(input_dim, dict_dim) * 0.1)

    def soft_threshold(self, H: torch.Tensor, threshold: float) -> torch.Tensor:
        """Applies proximal operator: sign(H) * max(0, |H| - threshold)."""
        return torch.sign(H) * torch.relu(torch.abs(H) - threshold)

    def forward(
        self, 
        z: torch.Tensor, 
        stage: sb.Stage = sb.Stage.TRAIN,
        wav_lens: torch.Tensor = None,
    ) -> tuple:
        B, D_in, T = z.shape

        # 1. Normalize dictionary columns
        W_norm = torch.linalg.vector_norm(
            self.W, ord=2, dim=0, keepdim=True
        ).clamp(min=1e-5)
        W_normalized = self.W / W_norm  # Shape: (D_in, K)

        W_cnt = W_normalized[:, : self.mid]  # (D_in, K_cnt)
        W_spk = W_normalized[:, self.mid :]  # (D_in, K_spk)

        # 2. Functional Initializations (NO prototype table / NO discrete IDs)
        H_cnt = torch.randn(
            B, self.mid, T, device=W_cnt.device, dtype=W_cnt.dtype
        )  # (B, K_cnt, T)
        h_spk = torch.randn(
            B,
            self.dict_dim - self.mid,
            1,
            device=W_cnt.device,
            dtype=W_cnt.dtype,
        )  # (B, K_spk, 1)

        # Mask of valid (non-padded) frames, (B, 1, T)
        mask, n_valid = cnt_mask(z, wav_lens)
        H_cnt = H_cnt * mask

        # 3. Dynamic Lipschitz Step Size Calculation
        with torch.no_grad():
            s = torch.linalg.svdvals(W_normalized)[0]
            L = (s**2).item()
            eta = 1.0 / max(L, 1e-5)
            threshold = eta * self.l1_lambda

        # 4. ISTA Optimization Loop (Applies in ALL stages identically)
        for step in range(self.num_steps):
            z_cnt = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt)
            # Expand speaker code across frames
            H_spk = h_spk.expand(B, self.dict_dim - self.mid, T)
            z_spk = torch.einsum("dk, bkr -> bdr", W_spk, H_spk)
            
            residual = (z - (z_cnt + z_spk)) * mask

            # Content update: (B, K_cnt, T)
            grad_cnt = torch.einsum("dk, bdt -> bkt", W_cnt, residual)
            H_cnt = self.soft_threshold(H_cnt + eta * grad_cnt, threshold) * mask

            # Speaker update: average residual projection across T -> (B, K_spk, 1)
            grad_spk = torch.einsum("dk, bdt -> bk", W_spk, residual).unsqueeze(-1) / n_valid.unsqueeze(-1)
            h_spk = self.soft_threshold(h_spk + eta * grad_spk, threshold)

        # 5. Final Reconstruction & Regularization

        z_cnt = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt)
        z_spk = torch.einsum("dk, bkr -> bdr", W_spk, H_spk)
        z_approx = z_cnt + z_spk

        return z_approx, H_cnt, h_spk, H_spk


class ResidualSparseDisentangle(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dict_dim: int,
        num_sparse_layers: int = 4,
        num_steps: int = 20,
        content_ratio: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.num_layers = num_sparse_layers
        self.content_ratio = content_ratio
        self.num_steps = num_steps


        self.sparse_module_list = nn.ModuleList(
            [
                SparseDisentangle(
                    input_dim,
                    dict_dim,
                    content_ratio=content_ratio,
                    num_steps=num_steps,
                )
                for _ in range(num_sparse_layers)
            ]
        )

        self.mid = int(dict_dim * content_ratio)
        self.spk_dim = dict_dim - self.mid
        self.spk_concat_dim = self.spk_dim * num_sparse_layers
        self.cnt_concat_dim = self.mid * num_sparse_layers

    def forward(self, z: torch.Tensor, stage: sb.Stage = sb.Stage.TRAIN, wav_lens: torch.Tensor = None):
        residual = z

        all_h_cnt = []
        all_h_spk = []
        all_h_spk_mat = []

        total_reconstruction = 0.0
        total_l12_reg_content = 0.0
        total_l1_reg_speaker = 0.0

        for sparse_module in self.sparse_module_list:
            z_approx, H_cnt, h_spk, H_spk = sparse_module(residual, stage=stage, wav_lens=wav_lens)
            all_h_cnt.append(H_cnt)
            all_h_spk.append(h_spk)
            all_h_spk_mat.append(H_spk)

            residual = residual - z_approx
            total_reconstruction = total_reconstruction + z_approx

            # Compute L12 regularization for content and speaker features
            l12_content = l12_cnt_loss(H_cnt)

            total_l12_reg_content = total_l12_reg_content + l12_content
            l1_speaker = l1_spk_loss(h_spk)
            total_l1_reg_speaker = total_l1_reg_speaker + l1_speaker

        # Compute recosntruction loss
        recon_loss_val = recon_loss(total_reconstruction, z)

        # Aggregate layer-wise codes
        all_h_cnt = torch.cat(all_h_cnt, dim=1)
        all_h_spk = torch.cat(all_h_spk, dim=1)
        all_h_spk_mat = torch.cat(all_h_spk_mat, dim=1)
        all_h = torch.stack([all_h_cnt, all_h_spk_mat], dim=1)

        return recon_loss_val, total_l12_reg_content, total_l1_reg_speaker, all_h_cnt, all_h_spk, all_h_spk_mat, all_h
