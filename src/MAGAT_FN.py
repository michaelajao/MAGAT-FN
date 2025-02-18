import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# TUNABLE MODEL PARAMETERS (DEFAULTS)
# =============================================================================
DEFAULT_HIDDEN_DIM = 128
DEFAULT_ATTENTION_HEADS = 8
DEFAULT_ATTENTION_REG_WEIGHT = 1e-4
DEFAULT_DROPOUT = 0.45
DEFAULT_NUM_SCALES = 2
DEFAULT_KERNEL_SIZE = 3
DEFAULT_TEMP_CONV_OUT_CHANNELS = 32

# =============================================================================
# 1. Adaptive Graph Attention Module (AGAM)
# =============================================================================
class AdaptiveGraphAttentionModule(nn.Module):
    """Adaptive Graph Attention Module (AGAM) with learnable adjacency bias and attention regularization."""
    def __init__(self, hidden_dim, num_nodes, dropout=DEFAULT_DROPOUT, attn_heads=DEFAULT_ATTENTION_HEADS, attn_reg_weight=DEFAULT_ATTENTION_REG_WEIGHT):
        super(AdaptiveGraphAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = attn_heads  # Number of attention heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.attn_reg_weight = attn_reg_weight  # Weight for the attention regularization loss

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        # Learnable adjacency bias: one bias per head and per node pair.
        self.learnable_adj = Parameter(torch.Tensor(1, self.heads, num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.learnable_adj)

    def forward(self, x, mask=None):
        B, N, H = x.shape  # (batch, num_nodes, hidden_dim)
        # Linear projections
        q = self.query(x).view(B, N, self.heads, self.head_dim)
        k = self.key(x).view(B, N, self.heads, self.head_dim)
        v = self.value(x).view(B, N, self.heads, self.head_dim)
        # Transpose to shape: [B, heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Add learnable adjacency bias
        scores = scores + self.learnable_adj
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # Save attention weights for visualization
        self.attn = attn.detach().cpu()
        
        attn = self.dropout(attn)
        # Attention regularization loss (e.g., L1 sparsity)
        attn_reg_loss = self.attn_reg_weight * torch.mean(torch.abs(attn))
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        output = self.out_proj(out)
        return output, attn_reg_loss

# =============================================================================
# 2. Multi-scale Temporal Fusion Module (MTFM)
# =============================================================================
class MultiScaleTemporalFusionModule(nn.Module):
    """Multi-scale Temporal Fusion Module (MTFM) for adaptive fusion of multi-scale temporal features."""
    def __init__(self, hidden_dim, num_scales=DEFAULT_NUM_SCALES, dropout=DEFAULT_DROPOUT):
        super(MultiScaleTemporalFusionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Multi-scale convolutions with varying kernel sizes (2^i)
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2**i, padding=2**i // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        ])
        # Learnable weights for adaptive fusion (one per scale)
        self.fusion_weight = Parameter(torch.ones(num_scales), requires_grad=True)
        # Optional additional fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, N, H] -> transpose to [B, H, N] for conv1d
        x = x.transpose(1, 2)
        features = []
        for scale in self.scales:
            feat = scale(x)  # [B, H, N]
            if feat.size(-1) != x.size(-1):
                feat = F.interpolate(feat, size=x.size(-1), mode='linear', align_corners=False)
            features.append(feat)
        # Adaptive weighting: softmax over learnable weights
        alpha = F.softmax(self.fusion_weight, dim=0)  # [num_scales]
        # Stack features: [B, num_scales, H, N]
        stacked = torch.stack(features, dim=1)
        # Weighted sum over scales
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        # Transpose back to [B, N, H]
        fused = fused.transpose(1, 2)
        out = self.fusion(fused)
        return out

# =============================================================================
# 3. Progressive Prediction and Refinement Module (PPRM)
# =============================================================================
class ProgressivePredictionRefinementModule(nn.Module):
    """Progressive Prediction and Refinement Module (PPRM) that refines predictions via a gating mechanism."""
    def __init__(self, hidden_dim, horizon, dropout=DEFAULT_DROPOUT):
        super(ProgressivePredictionRefinementModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        # Predictor for initial forecast
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )
        # Lightweight refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
            nn.Sigmoid()
        )

    def forward(self, x, x_last):
        # x: [B, N, H] ; x_last: [B, N]
        preds = self.predictor(x)  # [B, N, horizon]
        gates = self.refine_gate(x)  # [B, N, horizon]
        # Blend last observation with initial prediction
        x_last = x_last.unsqueeze(-1).expand_as(preds)
        refined = gates * x_last + (1 - gates) * preds
        return refined

# =============================================================================
# 4. Overall Model: MAGAT-FN (Multi-scale Adaptive Graph Attention Temporal Fusion Network)
# =============================================================================
class MAGATFN(nn.Module):
    """MAGAT-FN: Multi-scale Adaptive Graph Attention Temporal Fusion Network for epidemic prediction."""
    def __init__(self, args, data):
        super(MAGATFN, self).__init__()
        self.m = data.m
        self.window = args.window
        self.horizon = args.horizon

        # Use tunable parameters from args if provided; otherwise use defaults.
        self.hidden_dim = getattr(args, 'hidden_dim', DEFAULT_HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', DEFAULT_KERNEL_SIZE)

        # Temporal convolution with fixed dilation
        self.temp_conv = nn.Sequential(
            nn.Conv1d(1, DEFAULT_TEMP_CONV_OUT_CHANNELS, kernel_size=self.kernel_size, padding=self.kernel_size // 2),
            nn.BatchNorm1d(DEFAULT_TEMP_CONV_OUT_CHANNELS),
            nn.ReLU(),
            nn.Dropout(getattr(args, 'dropout', DEFAULT_DROPOUT))
        )
        # Feature processing: flatten temporal features and reduce dimension
        self.feature_process = nn.Sequential(
            nn.Linear(DEFAULT_TEMP_CONV_OUT_CHANNELS * self.window, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        # Main components
        self.graph_attention = AdaptiveGraphAttentionModule(
            self.hidden_dim, num_nodes=self.m,
            dropout=getattr(args, 'dropout', DEFAULT_DROPOUT),
            attn_heads=getattr(args, 'attn_heads', DEFAULT_ATTENTION_HEADS)
        )
        self.temporal_fusion = MultiScaleTemporalFusionModule(
            self.hidden_dim,
            num_scales=getattr(args, 'num_scales', DEFAULT_NUM_SCALES),
            dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
        )
        self.prediction_refinement = ProgressivePredictionRefinementModule(
            self.hidden_dim, self.horizon,
            dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
        )

    def forward(self, x, idx=None):
        # x: [B, T, N]
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last time step: [B, N]
        # Reshape for temporal convolution: [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temp_features = self.temp_conv(x_temp)  # [B*N, Channels, T]
        temp_features = temp_features.view(B, N, -1)  # [B, N, Channels * T]
        # Process features
        features = self.feature_process(temp_features)  # [B, N, hidden_dim]
        # Apply adaptive graph attention (returns output and auxiliary reg loss)
        graph_features, attn_reg_loss = self.graph_attention(features)
        # Multi-scale temporal fusion
        fusion_features = self.temporal_fusion(graph_features)
        # Progressive prediction and refinement (using last observation)
        predictions = self.prediction_refinement(fusion_features, x_last)  # [B, N, horizon]
        predictions = predictions.transpose(1, 2)  # [B, horizon, N]
        return predictions, attn_reg_loss
