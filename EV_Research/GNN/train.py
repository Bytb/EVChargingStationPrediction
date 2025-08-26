import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from Models.GCN import GCN
from Data import features, edges, labels, DEVICE
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
from tqdm import tqdm
from networkx.convert_matrix import to_scipy_sparse_array
from networkx import to_scipy_sparse_array
import scipy.sparse as sp


# -------- SET RANDOM SEED --------
SEED = 124
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------- Build Graph --------
G = nx.from_pandas_edgelist(edges)
valid_nodes = set(features.index) & set(labels.index)
G = G.subgraph(valid_nodes).copy()
node_list = sorted(G.nodes())
features = features.loc[node_list]
labels = labels.loc[node_list]

# -------- Tensors --------
x = torch.tensor(features.values, dtype=torch.float32).to(DEVICE)
y_raw = labels.values.squeeze()

# -------- Masks --------
num_nodes = len(node_list)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
indices = torch.randperm(num_nodes)
train_cutoff = int(0.6 * num_nodes)
val_cutoff = int(0.8 * num_nodes)
train_mask[indices[:train_cutoff]] = True
val_mask[indices[train_cutoff:val_cutoff]] = True
test_mask[indices[val_cutoff:]] = True

# -------- Normalize Labels --------
y_train_vals = y_raw[train_mask.numpy()]
label_mean = y_train_vals.mean()
label_std = y_train_vals.std() if y_train_vals.std() > 1e-8 else 1e-8
y_norm = (y_raw - label_mean) / label_std
y = torch.tensor(y_norm, dtype=torch.float32).to(DEVICE)

# -------- Adjacency Matrix --------
# --- Convert NX to scipy sparse ---
A_sp = to_scipy_sparse_array(G, nodelist=node_list, format="coo")

# --- Add self-loops ---
A_sp = A_sp + sp.eye(A_sp.shape[0])

# --- Normalize: D^(-1/2) A D^(-1/2) ---
rowsum = np.array(A_sp.sum(1)).flatten()
d_inv_sqrt = np.power(rowsum, -0.5)
d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
D_inv_sqrt = sp.diags(d_inv_sqrt)
A_norm = D_inv_sqrt @ A_sp @ D_inv_sqrt

# --- Convert to PyTorch sparse tensor ---
A_norm = A_norm.tocoo()
indices = torch.tensor([A_norm.row, A_norm.col], dtype=torch.long)
values = torch.tensor(A_norm.data, dtype=torch.float32)
A = torch.sparse_coo_tensor(indices, values, A_norm.shape).to(DEVICE)

# -------- Training --------
LEARNING_RATE = 0.001
EPOCHS = 300

model = GCN(in_dim=x.size(1), out_dim=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

train_losses = []
val_losses = []

for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    optimizer.zero_grad()
    out = model(x, A).squeeze()
    train_loss = F.mse_loss(out[train_mask], y[train_mask])
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    train_losses.append(train_loss.item())

    model.eval()
    with torch.no_grad():
        val_out = model(x, A).squeeze()
        val_loss = F.mse_loss(val_out[val_mask], y[val_mask])
        val_losses.append(val_loss.item())

    #print(f"Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")


# -------- Evaluation --------
model.eval()
with torch.no_grad():
    preds = model(x, A).squeeze().cpu().numpy()
    print("Max/min of raw preds:", np.nanmax(preds), np.nanmin(preds))
    preds_denorm = preds * label_std + label_mean
    true_vals = y_raw
    print("NaNs in preds_denorm?", np.isnan(preds_denorm[test_mask.cpu().numpy()]).sum())
    print("NaNs in true_vals?", np.isnan(true_vals[test_mask.cpu().numpy()]).sum())

test_mse = mean_squared_error(true_vals[test_mask.cpu().numpy()], preds_denorm[test_mask.cpu().numpy()])
mae = mean_absolute_error(true_vals[test_mask.cpu().numpy()], preds_denorm[test_mask.cpu().numpy()])
print(f"Test MSE (denormalized): {test_mse:.4f}")
print(f"Test MAE (denormalized): {mae:.4f}")

# -------- Metrics Plot --------
fig, axes = plt.subplots(1, 3, figsize=(32, 6), gridspec_kw={'wspace': 0.2})

# Scatter Plot
axes[0].scatter(true_vals, preds_denorm, alpha=0.7)
axes[0].plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
axes[0].set_xlabel("True Value")
axes[0].set_ylabel("Predicted Value")
axes[0].set_title("True vs Predicted")
axes[0].grid(True)

# Loss Curve
axes[1].plot(train_losses, label="Train Loss")
axes[1].plot(val_losses, label="Val Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Training vs Validation Loss")
axes[1].legend()
axes[1].grid(True)

# Distribution Histogram
axes[2].hist(true_vals, bins=30, alpha=0.6, label="True", edgecolor='black')
axes[2].hist(preds_denorm, bins=30, alpha=0.6, label="Predicted", edgecolor='black')
axes[2].set_title("Label vs Prediction Distribution")
axes[2].set_xlabel("Value")
axes[2].set_ylabel("Frequency")
axes[2].legend()
axes[2].grid(True)

plt.show()
