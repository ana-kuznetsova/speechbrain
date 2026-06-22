import torch
import torch.nn.functional as F
import torch.nn as nn
import logging


class SpeakerStrategy(nn.Module):
    """Abstract base class to support different speaker strategies in modular way.
    """
    def build_train_speaker(
        self,
        h_stacked,
        mid,
        layer_mixer,
        speaker_codes=None,
        **kwargs,
    ):
        raise NotImplementedError

    def build_vc_speaker(self, h_target, mid, layer_mixer, target_length=None):
        raise NotImplementedError
    

class GlobalSpeakerStrategy(SpeakerStrategy):
    def __init__(self):
        super().__init__()
        self.speaker_frame_map = {}

    def _get_or_sample_frame(self, speaker_code, num_frames):
        speaker_code = int(speaker_code)
        if speaker_code not in self.speaker_frame_map:
            self.speaker_frame_map[speaker_code] = torch.randint(
                low=0, high=num_frames, size=(1,)
            ).item()
        # Protect against varying sequence lengths across batches.
        return self.speaker_frame_map[speaker_code] % num_frames

    def build_train_speaker(
        self,
        h_stacked,
        mid,
        layer_mixer,
        speaker_codes=None,
        **kwargs,
    ):
        h_spk_time = layer_mixer(h_stacked, mid, type="spk")
        B, T, D = h_spk_time.shape

        unit_vec = torch.ones(B, 1, T, device=h_spk_time.device, dtype=h_spk_time.dtype)
        
        if speaker_codes.ndim > 1 and speaker_codes.shape[-1] == 1:
            speaker_codes = speaker_codes.squeeze(-1)
        if speaker_codes.shape[0] != B:
            raise ValueError(
                f"speaker_codes batch size mismatch: expected {B}, got {speaker_codes.shape[0]}"
            )
        selected_frames = torch.tensor(
            [self._get_or_sample_frame(code, T) for code in speaker_codes.tolist()],
            device=h_spk_time.device,
            dtype=torch.long,
        )
        batch_idx = torch.arange(B, device=h_spk_time.device)
        h_spk_time = h_spk_time[batch_idx, selected_frames].unsqueeze(-1)
                                                 
        h_spk_global = h_spk_time * unit_vec

        h_spk_global = h_spk_global.permute(0, 2, 1).contiguous()
        return h_spk_global


class TimeVaryingAttentionSpeakerStrategy(SpeakerStrategy):
    def __init__(self, feature_dim, query_dim=None, num_heads=4, dropout=0.1):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )
        if query_dim is None:
            query_dim = feature_dim
        self.query_proj = (
            nn.Identity() if query_dim == feature_dim else nn.Linear(query_dim, feature_dim)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def _refine(self, h_spk_time, query_features=None, key_padding_mask=None):
        # If no explicit query is provided, fall back to self-attention behavior.
        query = h_spk_time if query_features is None else self.query_proj(query_features)
        attn_out, _ = self.attn(
            query,
            h_spk_time,
            h_spk_time,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(query + self.dropout(attn_out))

    def build_train_speaker(
        self,
        h_stacked,
        mid,
        layer_mixer,
        speaker_codes=None,
        **kwargs,
    ):
        h_spk_time = layer_mixer(h_stacked, mid, type="spk")
        query_features = kwargs.get("query_features", None)
        # Optional frame lengths can be passed via kwargs as [B] (counts or relative [0, 1]).
        frame_lens = kwargs.get("frame_lens", None)
        key_padding_mask = None
        if frame_lens is not None:
            max_t = h_spk_time.shape[1]
            frame_lens = frame_lens.to(h_spk_time.device)
            if torch.is_floating_point(frame_lens):
                frame_lens = (frame_lens * max_t).round().long()
            else:
                frame_lens = frame_lens.long()
            frame_lens = frame_lens.clamp(min=1, max=max_t)
            time_ids = torch.arange(max_t, device=h_spk_time.device).unsqueeze(0)
            key_padding_mask = time_ids >= frame_lens.unsqueeze(1)
        return self._refine(
            h_spk_time,
            query_features=query_features,
            key_padding_mask=key_padding_mask,
        )

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
            h_layers = h_stacked[:, :, :, :mid]
        elif type == "spk":
            h_layers = h_stacked[:, :, :, mid:]
        else:
            raise ValueError("type must be 'cnt' or 'spk'")

        # 2. Concatenate layer-wise content features instead of summing.
        # [B, L, T, C] -> [B, T, L, C] -> [B, T, L*C]
        h_cnt_spk = h_layers.permute(0, 2, 1, 3).contiguous()
        h_cnt_spk = h_cnt_spk.view(h_cnt_spk.shape[0], h_cnt_spk.shape[1], -1)

        return h_cnt_spk

class SparseDisentangle(nn.Module):
    """
    Sparse decomposition module: learns a dictionary W and infers sparse codes h for input x,
    such that x ≈ W @ h. Enforces sparsity on h via L1 regularization.

    Args:
        input_dim: The dimension of the input representation.
        dict_dim: The number of dictionary atoms (columns in W).
    """

    def __init__(self, input_dim, dict_dim, content_ratio=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.content_ratio = content_ratio
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
            raise ValueError(
                "Input must be (B, D, T). Got shape: {}".format(orig_shape)
            )

        # Infer sparse codes
        h = self.h_encoder(x)  # (B*T, dict_dim)
        h = torch.relu(h)
        # Reconstruction
        x_approx = torch.matmul(h, self.W)  # (B*T, input_dim)

        # Subspace L1 penalties
        mid = int(h.shape[1] * self.content_ratio)
        l1_reg_content = h[:, :mid].abs().mean()
        l1_reg_speaker = h[:, mid:].abs().mean()
        sparse_loss = F.mse_loss(x_approx, x.reshape(-1, self.input_dim))

        # Reshape outputs to match input
        if len(shape_out) == 3:
            x_approx = x_approx.view(shape_out)
            h = h.view(shape_out[0], shape_out[1], self.dict_dim)
            if out_permute is not None:
                x_approx = x_approx.permute(out_permute)
                h = h.permute(out_permute[0], out_permute[2], out_permute[1])
        return x_approx, h, sparse_loss, l1_reg_content, l1_reg_speaker

'''
class ResidualSparseDisentangle(nn.Module):
    """Hierachical sparse disentanglement with residual connections. Stacks multiple SparseDisentangle modules, where each module tries to explain the residual left by the previous modules. This allows for a more flexible
    decomposition where different layers can capture different aspects of the signal, and the final output is the sum of all approximations.
    Args:
        input_dim: The dimension of the input representation.
        dict_dim: The number of dictionary atoms (columns in W) for each SparseDisentangle layer.
        num_sparse_layers: The number of SparseDisentangle layers to stack.
        content_ratio: The ratio of dictionary atoms responsible for content vs speaker information.
    Returns:
    - total_reconstruction: The sum of the approximations from all layers, shape (B, D, T)
    - h_stacked: The stacked sparse codes from all layers, shape (B, num_layers, dict_dim, T)
    - total_l1_reg_content: The sum of content L1 regularization across all layers (scalar)
    - total_l1_reg_speaker: The sum of speaker L1 regularization across all layers (scalar)
    """

    def __init__(self, input_dim, dict_dim, num_sparse_layers=4, content_ratio=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.num_layers = num_sparse_layers
        self.content_ratio = content_ratio
        self.sparse_module_list = nn.ModuleList(
            [SparseDisentangle(input_dim, dict_dim, content_ratio) for _ in range(num_sparse_layers)]
        )
        # Internal Projections
        mid = int(dict_dim * content_ratio)
        content_dim = dict_dim - mid
        self.content_concat_dim = content_dim * num_sparse_layers
        self.asr_proj = nn.Linear(self.content_concat_dim, input_dim)
        self.spk_proj = nn.Linear(self.content_concat_dim, input_dim)
        self.layer_mixer = SparseLayerMixer(num_sparse_layers)
        self.decoder_adapter = DecoderAdapter(
            self.content_concat_dim * 2, input_dim
        )  # For VC decoding

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
        mid = int(h_stacked.shape[-1] * self.content_ratio)

        # --- INTERNAL CONTENT PROJECTION ---
        # Weighted sum across layers for content, Pool Time for ASR
        h_cnt = self.layer_mixer(h_stacked, mid, type="cnt")  # [B, T, 32 * num_layers]
        z_proj_content = self.asr_proj(h_cnt)

        # --- INTERNAL SPEAKER PROJECTION ---
        # Sum across layers, Pool Time for Identity
        #h_spk = h_stacked[:, :, :, mid:].mean(dim=1)  # [B, T, 32]
        h_spk = self.layer_mixer(h_stacked, mid, type="spk")  # [B, T, 32 * num_layers]
        z_proj_speaker = self.spk_proj(h_spk)
        h_projected = torch.cat([h_cnt, h_spk], dim=-1)
        h_projected = self.decoder_adapter(h_projected)  # [B, T, input_dim]
        adapter_loss = F.mse_loss(h_projected, z.permute(0, 2, 1).contiguous())  # Match input shape for loss

        return (
            z_proj_content,
            z_proj_speaker,
            h_stacked,
            h_projected,
            total_sparse_loss,
            total_l1_reg_content,
            total_l1_reg_speaker,
            adapter_loss
        )
    def vc_decode(self, h_source: torch.Tensor, h_target: torch.Tensor) -> torch.Tensor:
        """
        h_source/target: [B, num_layers, dict_dim, T]
        """
        # Assuming dict_dim is the dimension being sliced
        mid = int(h_source.shape[-1] * self.content_ratio)
        
        # 1. Extract Content from Source (Keep temporal resolution)
        h_cnt_source = self.layer_mixer(h_source, mid)
    
        # 2. Extract Speaker from Target with the same layer-mixing strategy used in forward
        h_spk_target = self.layer_mixer(h_target, mid, type="spk")
        
        # Global average pooling over time to get the "identity"
        #h_spk_target_global = h_spk_target.mean(dim=-2, keepdim=True)
        
        # 3. Align Speaker to Source Time
        # Broadcast the single target speaker vector to every frame of the source
        #T_src = h_cnt_source.shape[-2]
        #h_spk_tiled = h_spk_target_global.expand(-1, T_src, -1)
        
        # 4. Final Recombination
        # Concatenate along the dictionary dimension

        h_projected = torch.cat([h_cnt_source, h_spk_target], dim=-1)
        h_projected = self.decoder_adapter(h_projected)
        return h_projected
'''

class ResidualSparseDisentangle(nn.Module):
    def __init__(
        self,
        input_dim,
        dict_dim,
        num_sparse_layers=4,
        content_ratio=0.5,
        speaker_strategy="global",
        speaker_num_heads=4,
        speaker_attn_dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_dim = dict_dim
        self.num_layers = num_sparse_layers
        self.content_ratio = content_ratio
        self.speaker_strategy_name = speaker_strategy

        self.sparse_module_list = nn.ModuleList(
            [
                SparseDisentangle(input_dim, dict_dim, content_ratio)
                for _ in range(num_sparse_layers)
            ]
        )

        mid = int(dict_dim * content_ratio)
        spk_dim = dict_dim - mid
        self.spk_concat_dim = spk_dim * num_sparse_layers
        self.cnt_concat_dim = mid * num_sparse_layers

        self.layer_mixer = SparseLayerMixer(num_sparse_layers)
        self.asr_proj = nn.Linear(self.cnt_concat_dim, input_dim)
        self.spk_proj = nn.Linear(self.spk_concat_dim, input_dim)

        if speaker_strategy == "global":
            self.speaker_strategy = GlobalSpeakerStrategy()
        elif speaker_strategy == "time_varying_attention":
            self.speaker_strategy = TimeVaryingAttentionSpeakerStrategy(
                feature_dim=self.spk_concat_dim,
                query_dim=self.cnt_concat_dim,
                num_heads=speaker_num_heads,
                dropout=speaker_attn_dropout,
            )
        else:
            raise ValueError(
                f"Unknown speaker_strategy: {speaker_strategy}"
            )

        self.decoder_adapter = DecoderAdapter(
            self.cnt_concat_dim + self.spk_concat_dim,
            input_dim,
        )

    def _build_content_features(self, h_stacked, mid):
        return self.layer_mixer(h_stacked, mid, type="cnt")

    def _build_speaker_features(
        self,
        h_stacked,
        mid,
        speaker_codes=None,
        query_features=None,
        **kwargs,
    ):
        return self.speaker_strategy.build_train_speaker(
            h_stacked,
            mid,
            self.layer_mixer,
            speaker_codes=speaker_codes,
            query_features=query_features,
            **kwargs,
        )
    
    def forward(self, z, speaker_codes=None, **kwargs):
        residual = z
        all_h = []
        total_reconstruction = 0
        total_l1_reg_content = 0.0
        total_l1_reg_speaker = 0.0
        total_sparse_loss = 0.0

        for sparse_module in self.sparse_module_list:
            x_approx_i, h_i, sparse_loss_i, l1_cnt_i, l1_spk_i = sparse_module(residual)
            residual = residual - x_approx_i
            total_reconstruction = total_reconstruction + x_approx_i
            all_h.append(h_i)
            total_l1_reg_content += l1_cnt_i
            total_l1_reg_speaker += l1_spk_i
            total_sparse_loss += sparse_loss_i

        h_stacked = torch.stack(all_h, dim=1)
        mid = int(h_stacked.shape[-1] * self.content_ratio)

        h_cnt = self._build_content_features(h_stacked, mid)
        h_spk = self._build_speaker_features(
            h_stacked,
            mid,
            speaker_codes=speaker_codes,
            query_features=h_cnt,
            **kwargs,
        )

        z_proj_content = self.asr_proj(h_cnt)
        z_proj_speaker = self.spk_proj(h_spk)

        h_projected = torch.cat([h_cnt, h_spk], dim=-1)
        h_projected = self.decoder_adapter(h_projected)
        adapter_loss = F.mse_loss(
            h_projected,
            z.permute(0, 2, 1).contiguous(),
        )

        return (
            z_proj_content,
            z_proj_speaker,
            h_stacked,
            h_projected,
            total_sparse_loss,
            total_l1_reg_content,
            total_l1_reg_speaker,
            adapter_loss,
        )
