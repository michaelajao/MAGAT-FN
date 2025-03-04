"""
MAGAT-FN Ablation Study Components

This module contains alternative implementations of MAGAT-FN's key components
for ablation studies. It includes:

1. GraphConvolutionalLayer: Standard GCN layer (ablation for AGAM)
2. SingleScaleTemporalConvolution: Single-scale convolution (ablation for MTFM)
3. DirectMultiStepPrediction: Direct prediction (ablation for PPRM)
4. MAGATFN_Ablation: Model class with configurable components for ablation studies

These components are used to systematically evaluate the contribution of each
module to the overall model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# Import the original modules for reference and reuse
from MAGAT_FN import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_ATTENTION_REG_WEIGHT,
    DEFAULT_DROPOUT,
    DEFAULT_NUM_SCALES,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_TEMP_CONV_OUT_CHANNELS,
    AdaptiveGraphAttentionModule,
    MultiScaleTemporalFusionModule,
    ProgressivePredictionRefinementModule
)

# =============================================================================
# 1. Standard GCN Layer (for AGAM ablation)
# =============================================================================
class GraphConvolutionalLayer(nn.Module):
    """Standard GCN layer with fixed adjacency matrix.
    
    Used as an ablation replacement for the Adaptive Graph Attention Module (AGAM).
    This layer performs standard graph convolution without adaptive attention mechanisms.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        num_nodes (int): Number of nodes in the graph
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim), (batch_size, num_nodes, num_nodes)
        - Output: (batch_size, num_nodes, hidden_dim), scalar_loss
    """
    def __init__(self, hidden_dim, num_nodes, dropout=DEFAULT_DROPOUT):
        super(GraphConvolutionalLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, adj_matrix=None, mask=None):
        """Forward pass of the GCN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            adj_matrix (torch.Tensor, optional): Adjacency matrix. Defaults to None.
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            tuple: (output tensor of shape (batch_size, num_nodes, hidden_dim), attention regularization loss)
        """
        # x: [B, N, H], adj_matrix: [N, N] or [B, N, N]
        batch_size = x.size(0)
        
        # Default to identity matrix if no adjacency matrix is provided
        if adj_matrix is None:
            adj_matrix = torch.eye(self.num_nodes, device=x.device)
        
        # Ensure adj_matrix is batched if not already
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Apply GCN operation: X' = AXW
        x = self.linear1(x)
        x = torch.bmm(adj_matrix, x)  # Aggregate neighbor features
        x = F.relu(x)                  # Apply non-linearity
        x = self.dropout(x)            # Apply dropout for regularization
        x = self.linear2(x)            # Second linear transformation
        x = self.norm(x)               # Layer normalization
        
        # Return zero for attention regularization loss (not used in GCN)
        return x, 0.0

# =============================================================================
# 2. Single-scale Temporal Convolution (for MTFM ablation)
# =============================================================================
class SingleScaleTemporalConvolution(nn.Module):
    """Single-scale temporal convolution for ablation study.
    
    Used as an ablation replacement for the Multi-scale Temporal Fusion Module (MTFM).
    This layer processes temporal features at a single scale with a fixed kernel size.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim)
        - Output: (batch_size, num_nodes, hidden_dim)
    """
    def __init__(self, hidden_dim, dropout=DEFAULT_DROPOUT):
        super(SingleScaleTemporalConvolution, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Single-scale convolution with standard kernel size
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward pass of the single-scale temporal convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, hidden_dim)
        """
        # x: [B, N, H] -> transpose to [B, H, N] for conv1d
        x = x.transpose(1, 2)  # Conv1d expects channels dimension second
        
        # Apply single-scale convolution
        feat = self.conv(x)  # [B, H, N]
        
        # Transpose back to [B, N, H]
        feat = feat.transpose(1, 2)
        
        # Apply output projection
        out = self.fusion(feat)
        
        return out

# =============================================================================
# 3. Direct Multi-step Prediction (for PPRM ablation)
# =============================================================================
class DirectMultiStepPrediction(nn.Module):
    """Direct multi-step prediction for ablation study.
    
    Used as an ablation replacement for the Progressive Prediction and Refinement Module (PPRM).
    This layer directly predicts all future time steps without any refinement mechanism.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        horizon (int): Prediction horizon
        dropout (float, optional): Dropout rate. Defaults to DEFAULT_DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim), (batch_size, num_nodes)
        - Output: (batch_size, num_nodes, horizon)
    """
    def __init__(self, hidden_dim, horizon, dropout=DEFAULT_DROPOUT):
        super(DirectMultiStepPrediction, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Direct predictor for all forecast steps
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x, x_last):
        """Forward pass of the direct multi-step prediction module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            x_last (torch.Tensor): Last observation tensor of shape (batch_size, num_nodes)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, horizon)
        """
        # x: [B, N, H] ; x_last: [B, N]
        # Simple direct prediction (without refinement)
        preds = self.predictor(x)  # [B, N, horizon]
        return preds

# =============================================================================
# 4. Overall Model: MAGAT-FN with Ablation Options
# =============================================================================
class MAGATFN_Ablation(nn.Module):
    """MAGAT-FN: Multi-scale Adaptive Graph Attention Temporal Fusion Network with ablation options.
    
    This model allows for systematic ablation studies by replacing key components with simpler alternatives.
    
    Args:
        args (Namespace): Arguments containing model hyperparameters
        data (Dataset): Dataset object containing input data
    
    Shape:
        - Input: (batch_size, window, num_nodes)
        - Output: (batch_size, horizon, num_nodes), scalar_loss
    """
    def __init__(self, args, data):
        super(MAGATFN_Ablation, self).__init__()
        self.m = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.ablation = getattr(args, 'ablation', 'none')

        # Use tunable parameters from args if provided; otherwise use defaults.
        self.hidden_dim = getattr(args, 'hidden_dim', DEFAULT_HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', DEFAULT_KERNEL_SIZE)

        # Store adjacency matrix for GCN if needed
        self.adj_matrix = None
        if hasattr(data, 'adj'):
            self.adj_matrix = data.adj

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
        
        # Ablation-specific component initialization
        if self.ablation == 'no_agam':
            # Study 1.1: Replace Adaptive Graph Attention with standard GCN
            self.graph_module = GraphConvolutionalLayer(
                self.hidden_dim, num_nodes=self.m,
                dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
            )
        else:
            # Original: Adaptive Graph Attention Module
            self.graph_module = AdaptiveGraphAttentionModule(
                self.hidden_dim, num_nodes=self.m,
                dropout=getattr(args, 'dropout', DEFAULT_DROPOUT),
                attn_heads=getattr(args, 'attn_heads', DEFAULT_ATTENTION_HEADS)
            )
        
        if self.ablation == 'no_mtfm':
            # Study 1.2: Replace Multi-scale Temporal Fusion with single-scale
            self.temporal_module = SingleScaleTemporalConvolution(
                self.hidden_dim,
                dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
            )
        else:
            # Original: Multi-scale Temporal Fusion Module
            self.temporal_module = MultiScaleTemporalFusionModule(
                self.hidden_dim,
                num_scales=getattr(args, 'num_scales', DEFAULT_NUM_SCALES),
                dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
            )
        
        if self.ablation == 'no_pprm':
            # Study 1.3: Replace Progressive Prediction with direct multi-step
            self.prediction_module = DirectMultiStepPrediction(
                self.hidden_dim, self.horizon,
                dropout=getattr(args, 'dropout', DEFAULT_DROPOUT)
            )
        else:
            # Original: Progressive Prediction and Refinement Module
            self.prediction_module = ProgressivePredictionRefinementModule(
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
        # x: [B, T, N]
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last time step: [B, N]
        # Reshape for temporal convolution: [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temp_features = self.temp_conv(x_temp)  # [B*N, Channels, T]
        temp_features = temp_features.view(B, N, -1)  # [B, N, Channels * T]
        # Process features
        features = self.feature_process(temp_features)  # [B, N, hidden_dim]
        
        # Apply graph module (AGAM or GCN)
        graph_features, attn_reg_loss = self.graph_module(features, self.adj_matrix)
        
        # Apply temporal module (MTFM or single-scale)
        fusion_features = self.temporal_module(graph_features)
        
        # Apply prediction module (PPRM or direct multi-step)
        predictions = self.prediction_module(fusion_features, x_last)  # [B, N, horizon]
        predictions = predictions.transpose(1, 2)  # [B, horizon, N]
        
        return predictions, attn_reg_loss