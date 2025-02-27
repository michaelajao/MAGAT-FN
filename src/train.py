import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import logging
import random
import shutil  # For backing up best model checkpoint
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from math import sqrt
from scipy.stats import pearsonr
import pandas as pd  # For handling date ranges and saving CSV files

# Add parent directory to path (if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import our model (ensure the file is renamed to MAGAT_FN.py)
from MAGAT_FN import MAGATFN

from data import DataBasicLoader
import argparse
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set up directory for saving figures (report/figures)
figures_dir = os.path.join(parent_dir, "report", "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='region785')
parser.add_argument('--sim_mat', type=str, default='region-adj')
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--train', type=float, default=0.5)
parser.add_argument('--val', type=float, default=0.2)
parser.add_argument('--test', type=float, default=0.3)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='save')
parser.add_argument('--mylog', action='store_false', default=True)
parser.add_argument('--extra', type=str, default='')
parser.add_argument('--label', type=str, default='')
parser.add_argument('--pcc', type=str, default='')
parser.add_argument('--result', type=int, default=0)
parser.add_argument('--record', type=str, default='')
# New argument: starting date for forecast visualization (assume weekly frequency)
parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for forecast visualization')
args = parser.parse_args()
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = args.cuda and torch.cuda.is_available()
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
logger.info('cuda %s', args.cuda)

# Updated log_token (removed ablation parameter)
log_token = '%s.w-%s.h-%s' % (args.dataset, args.window, args.horizon)

if args.mylog:
    tensorboard_log_dir = os.path.join('tensorboard', log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

data_loader = DataBasicLoader(args)

# Instantiate the model
model = MAGATFN(args, data_loader)
logger.info('model %s', model)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:', pytorch_total_params)

def evaluate(data_loader, data, tag='val', show=0):
    model.eval()
    total_loss = 0.
    n_samples = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss_train = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m)

        x_value_mx.append(X.data.cpu())
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_pred_mx = y_pred_mx[:, -1, :]

    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min

    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    if not args.pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            pcc_tmp.append(pearsonr(y_true_states[:, k], y_pred_states[:, k])[0])
        pcc_states = np.mean(np.array(pcc_tmp))
    else:
        pcc_states = 1
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    y_true_flat = np.reshape(y_true_states, (-1))
    y_pred_flat = np.reshape(y_pred_states, (-1))
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    if show == 1:
        print('x value', x_value_states)
        print('ground true', y_true_flat.shape)
        print('predict value', y_pred_flat.shape)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5))) / 1e7
    if not args.pcc:
        pcc = pearsonr(y_true_flat, y_pred_flat)[0]
    else:
        pcc = 1
        pcc_states = 1
    r2 = r2_score(y_true_flat, y_pred_flat, multioutput='uniform_average')
    var = explained_variance_score(y_true_flat, y_pred_flat, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    return float(total_loss / n_samples), mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae

def train_epoch(data_loader, data):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples)

def visualize_forecast(data_loader, model, save_path):
    """
    Visualizes the forecast versus ground truth for one sample and one node.
    The forecast and ground truth are unnormalized to the original scale.
    The x-axis uses dates based on the provided start_date (assumed weekly frequency).
    """
    model.eval()
    # Obtain one batch from the test set
    batch = next(data_loader.get_batches(data_loader.test, args.batch, False))
    X, Y = batch[0], batch[1]
    index = batch[2]
    output, _ = model(X, index)  # output shape: [B, horizon, N]
    
    # For visualization, pick the first sample and first node
    forecast = output[0, :, 0].detach().cpu().numpy()
    # Use the ground truth from Y (assuming a single value per sample)
    ground_truth = Y[0, 0].detach().cpu().numpy()
    
    # Unnormalize forecast and ground truth for node 0
    forecast = forecast * (data_loader.max[0] - data_loader.min[0]) + data_loader.min[0]
    ground_truth = ground_truth * (data_loader.max[0] - data_loader.min[0]) + data_loader.min[0]
    
    # Generate a date range for the forecast horizon using start_date argument
    forecast_dates = pd.date_range(start=args.start_date, periods=args.horizon, freq='W')
    
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, forecast, marker='o', label='Forecast')
    plt.plot(forecast_dates, [ground_truth] * args.horizon, linestyle='--', label='Ground Truth')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecast vs Ground Truth (Sample 0, Node 0)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_matrices(data_loader, model, save_path):
    """
    Creates a three-panel figure comparing:
      (a) Geolocation Matrix,
      (b) Input Correlation Matrix, and
      (c) Learned Attention Matrix.
    """
    # (a) Geolocation Matrix
    if hasattr(data_loader, 'adj'):
        geo_mat = data_loader.adj.cpu().numpy()
    else:
        geo_mat = None

    # (b) Input Correlation Matrix computed from raw input data
    raw_data = data_loader.rawdat  # shape: [n_samples, num_nodes]
    input_corr = np.corrcoef(raw_data.T)

    # (c) Learned Attention Matrix: run one forward pass and retrieve stored attention
    batch = next(data_loader.get_batches(data_loader.test, args.batch, False))
    X, _, _ = batch
    _ = model(X, None)  # Forward pass; ensure that AGAM stores its attention in self.attn
    if hasattr(model.graph_attention, 'attn'):
        attn_mat = model.graph_attention.attn[0].detach().cpu().numpy()  # shape: (heads, N, N)
        attn_mat = np.mean(attn_mat, axis=0)
    else:
        attn_mat = None

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if geo_mat is not None:
        im0 = axes[0].imshow(geo_mat, cmap='viridis')
        axes[0].set_title("Geolocation Matrix")
        plt.colorbar(im0, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, 'No Geolocation Matrix', horizontalalignment='center')
        axes[0].set_title("Geolocation Matrix")

    im1 = axes[1].imshow(input_corr, cmap='viridis')
    axes[1].set_title("Input Correlation Matrix")
    plt.colorbar(im1, ax=axes[1])

    if attn_mat is not None:
        im2 = axes[2].imshow(attn_mat, cmap='viridis')
        axes[2].set_title("Learned Attention Matrix")
        plt.colorbar(im2, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, 'No Attention Matrix', horizontalalignment='center')
        axes[2].set_title("Learned Attention Matrix")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    
def visualize_node_level_forecasts(data_loader, model, save_path, node_indices=[0,1,2,3]):
    """
    For one test sample, plot forecast vs. ground truth for several selected nodes.
    Each subplot corresponds to one node.
    """
    model.eval()
    batch = next(data_loader.get_batches(data_loader.test, args.batch, False))
    X, Y = batch[0], batch[1]
    output, _ = model(X, None)  # output: [B, horizon, N]
    # Select the first sample
    sample_forecast = output[0].detach().cpu().numpy()  # shape: [horizon, N]
    sample_truth = Y[0].detach().cpu().numpy()            # shape: [N]
    
    # Unnormalize for each node (assume data_loader.max and min are arrays of length m)
    unnorm_forecast = sample_forecast.copy()
    unnorm_truth = sample_truth.copy()
    for i in range(len(unnorm_truth)):
        unnorm_forecast[:, i] = unnorm_forecast[:, i] * (data_loader.max[i] - data_loader.min[i]) + data_loader.min[i]
        unnorm_truth[i] = unnorm_truth[i] * (data_loader.max[i] - data_loader.min[i]) + data_loader.min[i]

    # Create subplots for selected nodes
    num_nodes = len(node_indices)
    fig, axes = plt.subplots(1, num_nodes, figsize=(5*num_nodes, 4))
    if num_nodes == 1:
        axes = [axes]
    for j, node in enumerate(node_indices):
        axes[j].plot(range(1, args.horizon+1), unnorm_forecast[:, node], marker='o', label='Forecast')
        axes[j].plot(range(1, args.horizon+1), [unnorm_truth[node]]*args.horizon, linestyle='--', label='Ground Truth')
        axes[j].set_title(f'Node {node}')
        axes[j].set_xlabel('Horizon')
        axes[j].set_ylabel('Value')
        axes[j].legend()
        axes[j].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# Training loop
# For tracking loss curves
train_losses = []
val_losses = []

bad_counter = 0
best_epoch = 0
best_val = 1e+20

try:
    print('Begin training')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(data_loader, data_loader.train)
        val_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = os.path.join(args.save_dir, '{}.pt'.format(log_token))
            torch.save(model.state_dict(), model_path)
            # Backup the best model checkpoint
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            shutil.copy(model_path, best_model_path)
            print('Best validation epoch:', epoch, time.ctime())
            test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.test, tag='test')
            print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(
                mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch', epoch)

# Plot and save the loss curves
fig_loss = plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
loss_fig_path = os.path.join(figures_dir, f"loss_curve_{log_token}.png")
plt.savefig(loss_fig_path, dpi=300)
plt.close(fig_loss)
logger.info("Loss curve saved to %s", loss_fig_path)

# Visualize forecast at original scale with date x-axis and save the figure
forecast_fig_path = os.path.join(figures_dir, f"forecast_{log_token}.png")
visualize_forecast(data_loader, model, forecast_fig_path)
logger.info("Forecast figure saved to %s", forecast_fig_path)

# Node-level forecast comparisons for selected nodes
node_forecast_fig_path = os.path.join(figures_dir, f"node_forecasts_{log_token}.png")
visualize_node_level_forecasts(data_loader, model, node_forecast_fig_path, node_indices=[0,1,2,3])
logger.info("Node-level forecast comparisons saved to %s", node_forecast_fig_path)

# Visualize matrices: Geolocation, Input Correlation, and Learned Attention
matrices_fig_path = os.path.join(figures_dir, f"matrices_{log_token}.png")
visualize_matrices(data_loader, model, matrices_fig_path)
logger.info("Matrices comparison figure saved to %s", matrices_fig_path)

# Load the best model for final evaluation and print final metrics
model_path = os.path.join(args.save_dir, '{}.pt'.format(log_token))
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))
test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.test, tag='test', show=args.result)
print('Final evaluation')
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(
    mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))

# Save final evaluation metrics to a CSV file so that you don't have to retrain every time
results_dir = os.path.join(parent_dir, "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Prepare a dictionary with the desired metrics
final_metrics = {
    "MAE": [mae],
    "std_MAE": [std_mae],
    "RMSE": [rmse],
    "RMSEs": [rmse_states],
    "PCC": [pcc],
    "PCCs": [pcc_states],
    "MAPE": [mape],
    "R2": [r2],
    "R2s": [r2_states],
    "Var": [var],
    "Vars": [var_states],
    "Peak": [peak_mae]
}

# Create a DataFrame and save it as a CSV file
metrics_df = pd.DataFrame(final_metrics)
metrics_csv = os.path.join(results_dir, f"final_metrics_{log_token}.csv")
metrics_df.to_csv(metrics_csv, index=False)
logger.info("Saved final evaluation metrics to %s", metrics_csv)

if args.record != '':
    with open("result/result.txt", "a", encoding="utf-8") as f:
        f.write('Model: MAGATFN, dataset: {}, window: {}, horizon: {}, seed: {}, MAE: {:5.4f}, RMSE: {:5.4f}, PCC: {:5.4f}, lr: {}, dropout: {}\n'.format(
            args.dataset, args.window, args.horizon, args.seed, mae, rmse, pcc, args.lr, args.dropout))
