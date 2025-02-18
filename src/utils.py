import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error

def getLaplaceMat(batch_size, num_nodes, adj):
    """Get normalized Laplacian matrix."""
    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=adj.device)[None, :, :]
    
    # Calculate degree matrix
    degree = torch.sum(adj, dim=-1)  # [B, N]
    
    # Normalize adjacency matrix
    degree_inv_sqrt = torch.pow(degree + 1e-5, -0.5)
    degree_inv_sqrt = degree_inv_sqrt.unsqueeze(-1) * torch.eye(num_nodes, device=adj.device)[None, :, :]
    
    # Compute normalized Laplacian
    laplace = torch.matmul(torch.matmul(degree_inv_sqrt, adj), degree_inv_sqrt)
    
    return laplace

def peak_error(y_true_states, y_pred_states, threshold): 
    """Calculate mean absolute error in peak regions."""
    # Mask low values using threshold
    y_true_states[y_true_states < threshold] = 0
    mask_idx = np.argwhere(y_true_states <= threshold)
    for idx in mask_idx:
        y_pred_states[idx[0]][idx[1]] = 0
    
    # Calculate MAE only in peak regions
    peak_mae_raw = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    return peak_mae
