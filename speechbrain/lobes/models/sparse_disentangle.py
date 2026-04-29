import torch
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

    def __init__(self, input_dim, dict_dim, out_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.out_dim = out_dim if out_dim is not None else input_dim
        # Dictionary matrix W: shape (input_dim, dict_dim)
        self.W = nn.Parameter(torch.randn(input_dim, dict_dim) * 0.1)
        # Encoder to produce h from x: shape (input_dim) -> (dict_dim)
        self.h_encoder = nn.Linear(input_dim, dict_dim)

    def forward(self, x, l1_weight=1.0):
        """
        Args:
            x: Input tensor of shape (B, D, T) or (B, D) or (B, T, D)
            l1_weight: Weight for L1 regularization on h
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
                x_ = x.permute(0, 2, 1).contiguous().view(-1, self.input_dim)
                shape_out = (orig_shape[0], orig_shape[2], self.input_dim)
                out_permute = (0, 2, 1)
        else:
            raise ValueError("Input must be (B, D, T). Got shape: {}".format(orig_shape))

        # Infer sparse codes
        h = self.h_encoder(x_)  # (B*T, dict_dim)
        h = torch.relu(h)  # Enforce non-negativity (NMF style)
        # Reconstruction
        x_approx = torch.matmul(h, self.W.t())  # (B*T, input_dim)

        # L1 regularization on codes
        l1_reg_cont = l1_weight * torch.norm(h[:, :self.dict_dim // 2], p=1)
        l1_reg_sid = l1_weight * torch.norm(h[:, self.dict_dim // 2:], p=1)

        # Reshape outputs to match input
        if len(shape_out) == 3:
            x_approx = x_approx.view(shape_out)
            h = h.view(shape_out[0], shape_out[1], self.dict_dim)
            if out_permute is not None:
                x_approx = x_approx.permute(out_permute)
                h = h.permute(out_permute[0], out_permute[2], out_permute[1])
        return x_approx, h, l1_reg_cont, l1_reg_sid

class ResidualSparseDisentangle(nn.Module):
    """
    Residual version of SparseDisentangle: learns a dictionary W and infers sparse codes h for input x,
    such that x ≈ W @ h + x. Enforces sparsity on h via L1 regularization.

    Args:
        input_dim: The dimension of the input representation.
        dict_dim: The number of dictionary atoms (columns in W).
    """

    def __init__(self, input_dim, dict_dim):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        # Dictionary matrix W: shape (input_dim, dict_dim)
        self.W = nn.Parameter(torch.randn(input_dim, dict_dim) * 0.1)
        # Encoder to produce h from x: shape (input_dim) -> (dict_dim)
        self.h_encoder = nn.Linear(input_dim, dict_dim)

    def forward(self, x, l1_weight=1.0):
        """
        Args:
            x: Input tensor of shape (B, D, T) or (B, D) or (B, T, D)
            l1_weight: Weight for L1 regularization on h
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
                x_ = x.permute(0, 2, 1).contiguous().view(-1, self.input_dim)
                shape_out = (orig_shape[0], orig_shape[2], self.input_dim)
                out_permute = (0, 2, 1)
        else:
            raise ValueError("Input must be (B, D, T). Got shape: {}".format(orig_shape))

        # Infer sparse codes
        h = self.h_encoder(x_)  # (B*T, dict_dim)
        h = torch.relu(h)  # Enforce non-negativity (NMF style)
        # Reconstruction
        x_approx = torch.matmul(h, self.W.t()) + x_  # (