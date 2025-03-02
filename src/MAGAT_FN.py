"""
MAGAT-FN: Multi-scale Adaptive Graph Attention Temporal Fusion Network for Epidemic Forecasting

This module implements a deep learning model that combines graph attention mechanisms with
temporal fusion for epidemic prediction tasks. The model consists of three main components:

1. Adaptive Graph Attention Module (AGAM): Learns dynamic graph relationships
2. Multi-scale Temporal Fusion Module (MTFM): Processes multi-scale temporal patterns
3. Progressive Prediction and Refinement Module (PPRM): Refines predictions iteratively

The model is particularly suited for epidemic forecasting tasks where both spatial
relationships between regions and temporal patterns are important.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# TUNABLE MODEL PARAMETERS (DEFAULTS)
# =============================================================================
DEFAULT_HIDDEN_DIM = 128  # Dimension of hidden representations
DEFAULT_ATTENTION_HEADS = 8  # Number of parallel attention heads
DEFAULT_ATTENTION_REG_WEIGHT = 1e-4  # Weight for attention regularization
DEFAULT_DROPOUT = 0.45  # Dropout rate for regularization
DEFAULT_NUM_SCALES = 2  # Number of temporal scales in MTFM
DEFAULT_KERNEL_SIZE = 3  # Size of temporal convolution kernel
DEFAULT_TEMP_CONV_OUT_CHANNELS = 32  # Output channels for temporal convolution

# =============================================================================
# 1. Adaptive Graph Attention Module (AGAM)
# =============================================================================
class AdaptiveGraphAttentionModule(nn.Module):
    """Adaptive Graph Attention Module (AGAM) with learnable adjacency bias and attention regularization.
    
    This module implements a multi-head attention mechanism that learns dynamic spatial relationships
    between nodes in the graph. 

    
    Args:
        hidden_dim (int): Dimension of hidden representations
        num_nodes (int): Number of nodes in the graph
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
        attn_heads (int, optional): Number of attention heads. Defaults to DEFAULT_ATTENTION_HEADS
        attn_reg_weight (float, optional): Weight for attention regularization. Defaults to DEFAULT_ATTENTION_REG_WEIGHT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim)
        - Output: (batch_size, num_nodes, hidden_dim), scalar_loss
    """

    def __init__(self, hidden_dim, num_nodes, dropout=DEFAULT_DROPOUT, attn_heads=DEFAULT_ATTENTION_HEADS, 
                 attn_reg_weight=DEFAULT_ATTENTION_REG_WEIGHT):
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
        """Forward pass of the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None
            
        Returns:
            tuple: (output tensor of shape (batch_size, num_nodes, hidden_dim), attention regularization loss)
        """
        B, N, H = x.shape  # B: batch size, N: number of nodes, H: hidden dimension
        # Split hidden dim into multiple heads for parallel attention computation
        q = self.query(x).view(B, N, self.heads, self.head_dim)  # Project to query space
        k = self.key(x).view(B, N, self.heads, self.head_dim)    # Project to key space
        v = self.value(x).view(B, N, self.heads, self.head_dim)  # Project to value space
        
        # Rearrange for efficient batched matrix multiplication
        q = q.transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.transpose(1, 2)  # [B, heads, N, head_dim]
        v = v.transpose(1, 2)  # [B, heads, N, head_dim]
        
        # Compute scaled dot-product attention scores
        # Scale by sqrt(head_dim) to prevent exploding gradients
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add learnable graph structure bias to attention scores
        scores = scores + self.learnable_adj  # Bias helps capture persistent graph patterns
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply mask if provided
        
        # Normalize attention scores
        attn = F.softmax(scores, dim=-1)  # [B, heads, N, N]
        
        # Store attention weights for visualization/analysis
        self.attn = attn.detach().cpu()
        
        # Apply dropout for regularization
        attn = self.dropout(attn)
        
        # Compute attention regularization loss to encourage sparsity
        attn_reg_loss = self.attn_reg_weight * torch.mean(torch.abs(attn))
        
        # Apply attention weights to values
        out = torch.matmul(attn, v)  # Weighted aggregation of values
        
        # Restore original shape and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        output = self.out_proj(out)  # Final linear transformation
        
        return output, attn_reg_loss

# =============================================================================
# 2. Multi-scale Temporal Fusion Module (MTFM)
# =============================================================================
class MultiScaleTemporalFusionModule(nn.Module):
    """Multi-scale Temporal Fusion Module (MTFM) for adaptive fusion of multi-scale temporal features.
    
    This module processes temporal features at different scales using parallel dilated convolutions.

    
    Args:
        hidden_dim (int): Dimension of hidden representations
        num_scales (int, optional): Number of temporal scales. Defaults to DEFAULT_NUM_SCALES
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim)
        - Output: (batch_size, num_nodes, hidden_dim)
    """

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
        """Forward pass of the temporal fusion module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, hidden_dim)
        """
        # Transpose for temporal convolution: [B, N, H] -> [B, H, N]
        x = x.transpose(1, 2)  # Conv1d expects channels dimension second
        
        # Process input at different temporal scales
        features = []
        for scale in self.scales:
            feat = scale(x)  # Apply convolution at current scale
            # Ensure all scales have same sequence length via interpolation
            if feat.size(-1) != x.size(-1):
                feat = F.interpolate(feat, size=x.size(-1), 
                                   mode='linear', align_corners=False)
            features.append(feat)
        
        # Compute adaptive fusion weights
        alpha = F.softmax(self.fusion_weight, dim=0)  # [num_scales]
        
        # Stack and weight features from different scales
        stacked = torch.stack(features, dim=1)  # [B, num_scales, H, N]
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        
        # Restore original dimensions and apply final fusion
        fused = fused.transpose(1, 2)  # [B, N, H]
        out = self.fusion(fused)  # Additional processing
        return out

# =============================================================================
# 3. Progressive Prediction and Refinement Module (PPRM)
# =============================================================================
class ProgressivePredictionRefinementModule(nn.Module):
    """Progressive Prediction and Refinement Module (PPRM) that refines predictions via gating.
    
    This module implements iterative refinement of predictions, requiring multiple forward passes
    through neural networks per timestep. 
    Args:
        hidden_dim (int): Dimension of hidden representations
        horizon (int): Prediction horizon
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim), (batch_size, num_nodes)
        - Output: (batch_size, num_nodes, horizon)
    """

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
        """Forward pass of the prediction and refinement module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            x_last (torch.Tensor): Last observation tensor of shape (batch_size, num_nodes)
            
        Returns:
            torch.Tensor: Refined predictions of shape (batch_size, num_nodes, horizon)
        """
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
    """MAGAT-FN: Multi-scale Adaptive Graph Attention Temporal Fusion Network for epidemic prediction.
    
    This model combines graph attention mechanisms with multi-scale temporal fusion and progressive
    prediction refinement to predict epidemic trends. It processes input data through a series of
    convolutional, attention, and fusion layers to produce refined predictions.
    
    Args:
        args (Namespace): Arguments containing model hyperparameters
        data (Dataset): Dataset object containing input data
    
    Shape:
        - Input: (batch_size, window, num_nodes)
        - Output: (batch_size, horizon, num_nodes)
    """

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
        """Forward pass of the overall model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window, num_nodes)
            idx (torch.Tensor, optional): Index tensor. Defaults to None
            
        Returns:
            tuple: (predictions tensor of shape (batch_size, horizon, num_nodes), attention regularization loss)
        """
        # Extract dimensions and last observation
        B, T, N = x.shape  # B: batch size, T: sequence length, N: num nodes
        x_last = x[:, -1, :]  # Last timestep for each node: [B, N]
        
        # Reshape for temporal convolution
        # [B, T, N] -> [B*N, 1, T] for efficient batch processing
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Extract temporal patterns via convolution
        temp_features = self.temp_conv(x_temp)  # [B*N, Channels, T]
        temp_features = temp_features.view(B, N, -1)  # [B, N, Channels * T]
        
        # Process and compress temporal features
        features = self.feature_process(temp_features)  # [B, N, hidden_dim]
        
        # Apply graph attention to capture spatial dependencies
        graph_features, attn_reg_loss = self.graph_attention(features)
        
        # Fuse multi-scale temporal patterns
        fusion_features = self.temporal_fusion(graph_features)
        
        # Generate and refine predictions
        predictions = self.prediction_refinement(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)  # [B, horizon, N]
        
        return predictions, attn_reg_loss
