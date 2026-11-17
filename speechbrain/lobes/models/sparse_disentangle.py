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

    The options below were added to reduce speaker leakage into the content
    code. All of them default to the original behaviour.

    Arguments
    ---------
    input_dim : int
        Dimension of the input features z.
    dict_dim : int
        Total number of dictionary atoms (content + speaker).
    content_ratio : float
        Fraction of atoms assigned to the content branch.
    num_steps : int
        Number of unrolled ISTA/FISTA iterations.
    l1_lambda : float
        Sparsity weight. Absolute if ``relative_lambda`` is False, otherwise
        a fraction of the largest dictionary/target correlation (like the
        lambda_max normalisation of LASSO), which makes it independent of the
        scale of z.
    code_init : str
        "random" (original N(0, 1) init), "zeros", or "correlation"
        (soft-thresholded W^T z). The last two are deterministic and do not
        leave random noise in the codes after a few iterations.
    relative_lambda : bool
        Interpret ``l1_lambda`` relative to the data scale (see above).
    use_fista : bool
        Use FISTA momentum, which converges faster than plain ISTA for the
        same number of unrolled steps.
    mean_split : bool
        Split the target: the speaker branch only explains the per-utterance
        time-mean of z (solved on its own), and the content branch only
        explains the mean-removed signal. The content code then has no
        constant offset to absorb, which removes the main route for speaker
        information to enter it.
    speaker_first : bool
        Only used when ``mean_split`` is False. Update the speaker code first
        and give the content update the residual that already contains the
        new speaker code (Gauss-Seidel) instead of updating both blocks from
        the same residual (Jacobi), so the content block cannot double count
        what the speaker block has just explained.
    """

    def __init__(
        self,
        input_dim: int,
        dict_dim: int,
        content_ratio: float = 0.5,
        num_steps: int = 5,
        l1_lambda: float = 1e-4,
        code_init: str = "random",
        relative_lambda: bool = False,
        use_fista: bool = False,
        mean_split: bool = False,
        speaker_first: bool = False,
    ):
        super().__init__()
        if code_init not in ("random", "zeros", "correlation"):
            raise ValueError(
                f"code_init must be random, zeros or correlation, got {code_init}"
            )
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.content_ratio = content_ratio
        self.num_steps = num_steps
        self.l1_lambda = l1_lambda
        self.code_init = code_init
        self.relative_lambda = relative_lambda
        self.use_fista = use_fista
        self.mean_split = mean_split
        self.speaker_first = speaker_first
    
        # Split sizes
        self.mid = int(dict_dim * content_ratio)

        # Dictionary parameter W (input_dim x dict_dim)
        self.W = nn.Parameter(torch.randn(input_dim, dict_dim) * 0.1)

    def soft_threshold(self, H: torch.Tensor, threshold) -> torch.Tensor:
        """Applies proximal operator: sign(H) * max(0, |H| - threshold).

        ``threshold`` can be a float or a tensor broadcastable to H.
        """
        return torch.sign(H) * torch.relu(torch.abs(H) - threshold)

    def _normalized_dictionary(self) -> tuple:
        """Returns the unit-norm dictionary and its (content, speaker) blocks."""
        W_norm = torch.linalg.vector_norm(
            self.W, ord=2, dim=0, keepdim=True
        ).clamp(min=1e-5)
        W_normalized = self.W / W_norm  # Shape: (D_in, K)
        return W_normalized, W_normalized[:, : self.mid], W_normalized[:, self.mid :]

    def coherence_penalty(self) -> tuple:
        """Squared cosine similarity between atoms of the normalised dictionary.

        Returns
        -------
        cross : torch.Tensor
            Squared Frobenius norm of W_cnt^T W_spk. It is zero when the
            content and speaker subspaces are orthogonal, so neither branch
            can explain what the other one is meant to explain.
        intra : torch.Tensor
            Sum of squared off-diagonal entries of the Gram matrices of
            W_cnt and W_spk (within-group coherence), which discourages
            redundant atoms inside each branch.
        """
        _, W_cnt, W_spk = self._normalized_dictionary()
        cross = (W_cnt.t() @ W_spk).pow(2).sum()

        def off_diagonal_energy(W_block: torch.Tensor) -> torch.Tensor:
            gram = W_block.t() @ W_block
            return gram.pow(2).sum() - torch.diagonal(gram).pow(2).sum()

        intra = off_diagonal_energy(W_cnt) + off_diagonal_energy(W_spk)
        return cross, intra

    def forward(
        self, 
        z: torch.Tensor, 
        stage: sb.Stage = sb.Stage.TRAIN,
        wav_lens: torch.Tensor = None,
    ) -> tuple:
        B, D_in, T = z.shape
        K_spk = self.dict_dim - self.mid

        # 1. Normalize dictionary columns
        W_normalized, W_cnt, W_spk = self._normalized_dictionary()

        # Mask of valid (non-padded) frames, (B, 1, T)
        mask, n_valid = cnt_mask(z, wav_lens)

        # Masked per-utterance time-mean of z, (B, D_in, 1). This constant,
        # utterance-level component is where speaker/channel information mostly lives.
        z_mean = (z * mask).sum(dim=-1, keepdim=True) / n_valid.unsqueeze(-1)

        # 2. Targets each branch has to explain.
        if self.mean_split:
            # Speaker branch explains the time-mean only; the content branch
            # explains the mean-removed signal, so it has no constant offset
            # left to absorb (the main way speaker info leaked into H_cnt).
            target_cnt = (z - z_mean) * mask
            target_spk = z_mean
        else:
            # Original behaviour: both branches share the full signal.
            target_cnt = z * mask
            target_spk = z

        # 3. Dynamic Lipschitz step sizes and soft-thresholds
        with torch.no_grad():
            if self.mean_split:
                # The two blocks are independent problems, so each one gets
                # its own (larger) step size from its own Lipschitz constant.
                L_cnt = torch.linalg.svdvals(W_cnt)[0].item() ** 2
                L_spk = torch.linalg.svdvals(W_spk)[0].item() ** 2
            else:
                L_cnt = L_spk = torch.linalg.svdvals(W_normalized)[0].item() ** 2
            eta_cnt = 1.0 / max(L_cnt, 1e-5)
            eta_spk = 1.0 / max(L_spk, 1e-5)

            if self.relative_lambda:
                # lambda as a fraction of the smallest lambda that zeroes the
                # whole code (lambda_max), per utterance. The absolute value
                # 1e-4 was far below the scale of the codes, so nothing was
                # ever zeroed and the codes stayed ~100% dense.
                corr_cnt = (
                    torch.einsum("dk, bdt -> bkt", W_cnt, target_cnt)
                    .abs()
                    .amax(dim=(1, 2), keepdim=True)
                )
                corr_spk = (
                    torch.einsum("dk, bdr -> bkr", W_spk, z_mean)
                    .abs()
                    .amax(dim=(1, 2), keepdim=True)
                )
                threshold_cnt = eta_cnt * self.l1_lambda * corr_cnt  # (B, 1, 1)
                threshold_spk = eta_spk * self.l1_lambda * corr_spk  # (B, 1, 1)
            else:
                threshold_cnt = eta_cnt * self.l1_lambda
                threshold_spk = eta_spk * self.l1_lambda

        # 4. Code initialisation (NO prototype table / NO discrete IDs)
        if self.code_init == "random":
            H_cnt = torch.randn(
                B, self.mid, T, device=W_cnt.device, dtype=W_cnt.dtype
            )  # (B, K_cnt, T)
            h_spk = torch.randn(
                B, K_spk, 1, device=W_cnt.device, dtype=W_cnt.dtype
            )  # (B, K_spk, 1)
            H_cnt = H_cnt * mask
        elif self.code_init == "zeros":
            # Deterministic, no random noise left in the codes.
            H_cnt = torch.zeros(
                B, self.mid, T, device=W_cnt.device, dtype=W_cnt.dtype
            )
            h_spk = torch.zeros(
                B, K_spk, 1, device=W_cnt.device, dtype=W_cnt.dtype
            )
        else:
            # Warm start: one thresholded matched-filter step, W^T target.
            H_cnt = self.soft_threshold(
                torch.einsum("dk, bdt -> bkt", W_cnt, target_cnt), threshold_cnt
            ) * mask
            h_spk = self.soft_threshold(
                torch.einsum("dk, bdr -> bkr", W_spk, z_mean), threshold_spk
            )

        # 5. ISTA / FISTA Optimization Loop (Applies in ALL stages identically)
        # H_*_y are the points where gradients are evaluated; they equal the
        # current iterates unless FISTA momentum is enabled.
        H_cnt_y, h_spk_y = H_cnt, h_spk
        t_k = 1.0
        for step in range(self.num_steps):
            z_cnt_y = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt_y)
            z_spk_y = torch.einsum("dk, bkr -> bdr", W_spk, h_spk_y)  # (B, D, 1)

            # Speaker block is always updated first.
            if self.mean_split:
                # Speaker fits the time-mean only: (B, K_spk, 1)
                grad_spk = torch.einsum("dk, bdr -> bkr", W_spk, target_spk - z_spk_y)
            else:
                residual = (z - (z_cnt_y + z_spk_y)) * mask
                # Average residual projection across T -> (B, K_spk, 1)
                grad_spk = torch.einsum(
                    "dk, bdt -> bk", W_spk, residual
                ).unsqueeze(-1) / n_valid.unsqueeze(-1)
            h_spk_new = self.soft_threshold(h_spk_y + eta_spk * grad_spk, threshold_spk)

            # Content block update: (B, K_cnt, T)
            if self.mean_split:
                # Remainder after removing the time-mean: independent of h_spk.
                residual_cnt = (target_cnt - z_cnt_y) * mask
            elif self.speaker_first:
                # Gauss-Seidel: content sees the residual after the new
                # speaker code, so it does not re-explain the same component.
                z_spk_new = torch.einsum("dk, bkr -> bdr", W_spk, h_spk_new)
                residual_cnt = (z - (z_cnt_y + z_spk_new)) * mask
            else:
                # Original Jacobi update: both blocks use the same residual.
                residual_cnt = residual
            grad_cnt = torch.einsum("dk, bdt -> bkt", W_cnt, residual_cnt)
            H_cnt_new = (
                self.soft_threshold(H_cnt_y + eta_cnt * grad_cnt, threshold_cnt) * mask
            )

            if self.use_fista:
                # Nesterov momentum: extrapolate along the last update direction.
                t_next = (1.0 + (1.0 + 4.0 * t_k**2) ** 0.5) / 2.0
                momentum = (t_k - 1.0) / t_next
                H_cnt_y = H_cnt_new + momentum * (H_cnt_new - H_cnt)
                h_spk_y = h_spk_new + momentum * (h_spk_new - h_spk)
                t_k = t_next
            else:
                H_cnt_y, h_spk_y = H_cnt_new, h_spk_new
            H_cnt, h_spk = H_cnt_new, h_spk_new

        # 6. Final Reconstruction & Regularization
        # Built from the final codes (the original code reused a speaker
        # code that was one iteration old).
        H_spk = h_spk.expand(B, K_spk, T)
        z_cnt = torch.einsum("dk, bkt -> bdt", W_cnt, H_cnt)
        z_spk = torch.einsum("dk, bkr -> bdr", W_spk, H_spk)
        z_approx = z_cnt + z_spk

        return z_approx, H_cnt, h_spk, H_spk


class ResidualSparseDisentangle(nn.Module):
    """Stack of SparseDisentangle layers; each layer codes the residual left by
    the previous ones. The keyword options below are forwarded unchanged to
    every layer (see SparseDisentangle); defaults keep the original behaviour.
    """

    def __init__(
        self,
        input_dim: int,
        dict_dim: int,
        num_sparse_layers: int = 4,
        num_steps: int = 20,
        content_ratio: float = 0.5,
        l1_lambda: float = 1e-4,
        code_init: str = "random",
        relative_lambda: bool = False,
        use_fista: bool = False,
        mean_split: bool = False,
        speaker_first: bool = False,
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
                    l1_lambda=l1_lambda,
                    code_init=code_init,
                    relative_lambda=relative_lambda,
                    use_fista=use_fista,
                    mean_split=mean_split,
                    speaker_first=speaker_first,
                )
                for _ in range(num_sparse_layers)
            ]
        )

        self.mid = int(dict_dim * content_ratio)
        self.spk_dim = dict_dim - self.mid
        self.spk_concat_dim = self.spk_dim * num_sparse_layers
        self.cnt_concat_dim = self.mid * num_sparse_layers

    def coherence_penalty(self) -> tuple:
        """Dictionary coherence summed over layers.

        Returns (cross, intra): content-vs-speaker atom overlap and
        within-group atom overlap. See SparseDisentangle.coherence_penalty.
        """
        cross, intra = 0.0, 0.0
        for sparse_module in self.sparse_module_list:
            layer_cross, layer_intra = sparse_module.coherence_penalty()
            cross = cross + layer_cross
            intra = intra + layer_intra
        return cross, intra

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

        # Scale-free diagnostic (masked ||z - z_hat||^2 / ||z||^2) kept off the return
        # signature so other recipes that unpack the outputs keep working.
        with torch.no_grad():
            mask, _ = cnt_mask(z, wav_lens)
            err = ((z - total_reconstruction) * mask).pow(2).sum()
            energy = (z * mask).pow(2).sum()
            self.last_relative_recon_error = err / (energy + 1e-8)

        # Aggregate layer-wise codes
        all_h_cnt = torch.cat(all_h_cnt, dim=1)
        all_h_spk = torch.cat(all_h_spk, dim=1)
        all_h_spk_mat = torch.cat(all_h_spk_mat, dim=1)
        all_h = torch.stack([all_h_cnt, all_h_spk_mat], dim=1)

        return recon_loss_val, total_l12_reg_content, total_l1_reg_speaker, all_h_cnt, all_h_spk, all_h_spk_mat, all_h
