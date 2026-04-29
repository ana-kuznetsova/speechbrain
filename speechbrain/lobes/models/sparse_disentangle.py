import torch
import torch.nn.functional as F
import torch.nn as nn
import logging


class SparseDisentangle(nn.Module):
    """
    Sparse decomposition module: learns a dictionary W and infers sparse codes h for input x,
    such that x ≈ W @ h. Enforces sparsity on h via L1 regularization.

    Args:
        input_dim: The dimension of the input representation.
        dict_dim: The number of dictionary atoms (columns in W).
    """

    def __init__(self, input_dim, dict_dim):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        # Dictionary matrix W: shape (dict_dim, input_dim)
        self.W = nn.Parameter(torch.randn(dict_dim, input_dim) * 0.1)
        # Encoder to produce h from x: shape (input_dim) -> (dict_dim)
        self.h_encoder = nn.Linear(input_dim, dict_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, D, T) or (B, D) or (B, T, D)
        Returns:
            x_approx: Reconstruction (B, D, T) (matches input shape)
            h: Sparse codes (B, dict_dim, T)
            l1_reg: L1 regularization loss (scalar)
        """
        orig_shape = x.shape
        # Accept (B, D, T) or (B, T, D) or (B, D)
        if x.dim() == 3:
            # Assume (B, D, T) by default
            if orig_shape[1] == self.input_dim:
                # (B, D, T) -> (B*T, D)
                x = x.permute(0, 2, 1).contiguous().view(-1, self.input_dim)
                shape_out = (orig_shape[0], orig_shape[2], self.input_dim)
                out_permute = (0, 2, 1)
        else:
            raise ValueError("Input must be (B, D, T). Got shape: {}".format(orig_shape))

        # Infer sparse codes
        h = self.h_encoder(x)  # (B*T, dict_dim)
        h = torch.relu(h)
        # Reconstruction
        x_approx = torch.matmul(h, self.W)  # (B*T, input_dim)

         # Subspace L1 penalties
        mid = h.shape[1] // 2
        l1_reg_speaker = h[:, :mid].abs().mean()
        l1_reg_content = h[:, mid:].abs().mean()
        sparse_loss = F.mse_loss(x_approx, x.view(-1, self.input_dim))
        

        # Reshape outputs to match input
        if len(shape_out) == 3:
            x_approx = x_approx.view(shape_out)
            h = h.view(shape_out[0], shape_out[1], self.dict_dim)
            if out_permute is not None:
                x_approx = x_approx.permute(out_permute)
                h = h.permute(out_permute[0], out_permute[2], out_permute[1])
        return x_approx, h, sparse_loss, l1_reg_content, l1_reg_speaker

class ResidualSparseDisentangle(nn.Module):
    """Hierachical sparse disentanglement with residual connections. Stacks multiple SparseDisentangle modules, where each module tries to explain the residual left by the previous modules. This allows for a more flexible
    decomposition where different layers can capture different aspects of the signal, and the final output is the sum of all approximations.
    Args:
        input_dim: The dimension of the input representation.
        dict_dim: The number of dictionary atoms (columns in W) for each SparseDisentangle layer.
        num_sparse_layers: The number of SparseDisentangle layers to stack.
    Returns:
    - total_reconstruction: The sum of the approximations from all layers, shape (B, D, T)
    - h_stacked: The stacked sparse codes from all layers, shape (B, num_layers, dict_dim, T)
    - total_l1_reg_content: The sum of content L1 regularization across all layers (scalar)
    - total_l1_reg_speaker: The sum of speaker L1 regularization across all layers (scalar)
    """
    def __init__(self, input_dim, dict_dim, num_sparse_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.num_layers = num_sparse_layers
        self.sparse_module_list = nn.ModuleList([
            SparseDisentangle(input_dim, dict_dim) for _ in range(num_sparse_layers)
        ])
        # Internal Projections
        mid = dict_dim // 2
        self.asr_proj = nn.Linear(mid, input_dim) # Or keep original dim
        self.spk_proj = nn.Linear(mid, input_dim)

    def forward(self, z):
        """
        Args:
            z: Input tensor of shape (B, D, T)
        Returns:
            total_reconstruction: The sum of the approximations from all layers, shape (B, D, T)
            h_stacked: The stacked sparse codes from all layers, shape (B, num_layers, dict_dim, T)
            total_l1_reg_content: The sum of content L1 regularization across all layers (scalar)
            total_l1_reg_speaker: The sum of speaker L1 regularization across all layers (scalar)
        """
        residual = z
        all_h = []
        total_reconstruction = 0
        total_l1_reg_content = 0.0
        total_l1_reg_speaker = 0.0
        total_sparse_loss = 0.0
        
        for sparse_module in self.sparse_module_list:
            # x_approx: the "chunk" of signal explained by this layer
            x_approx_i, h_i, sparse_loss_i, l1_cnt_i, l1_spk_i = sparse_module(residual)
            
            # Update residual: Successive refinement
            residual = residual - x_approx_i
            
            # Accumulate for the final output
            total_reconstruction = total_reconstruction + x_approx_i
            all_h.append(h_i)
            
            total_l1_reg_content += l1_cnt_i
            total_l1_reg_speaker += l1_spk_i
            total_sparse_loss += sparse_loss_i
        # Stack H: Shape (B, num_layers, dict_dim, T) 
        # This allows you to pool across layers for the Speaker Head
        h_stacked = torch.stack(all_h, dim=1)
        mid = h_stacked.shape[-1] // 2

        # --- INTERNAL CONTENT PROJECTION ---
        # Mean across layers, keep Time for ASR
        h_cnt = h_stacked[:, :, :, mid:].mean(dim=1) # [B, T, 32]
        z_proj_content = self.asr_proj(h_cnt)

        # --- INTERNAL SPEAKER PROJECTION ---
        # Sum across layers, Pool Time for Identity
        h_spk = h_stacked[:, :, :, :mid].sum(dim=1)  # [B, T, 32]
        z_proj_speaker = self.spk_proj(h_spk)

        return z_proj_content, z_proj_speaker, h_stacked, total_sparse_loss, total_l1_reg_content, total_l1_reg_speaker
    