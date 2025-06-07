import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from GCN import GCN
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import mean_squared_error

# Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
edges = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Raw_Edges_Expanded.csv")
features = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Final_Joined_Features.csv", index_col=0)
labels = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Tampa_Labels_Test.csv", index_col=0)

# --- Format IDs ---
def adjust_edge(entry):
    road, year = entry.replace("Road", "").split("_")
    return f"Road{int(road)-1}_{year}"

edges['source'] = edges['source'].apply(adjust_edge)
edges['target'] = edges['target'].apply(adjust_edge)

def adjust_road_id(entry):
    road, year = entry.replace("Road", "").split("_")
    return f"Road{int(road)-1}_{year}"

features.index = features.index.to_series().apply(adjust_road_id)

# --- Normalize Features ---
def normalize_features(features, log_transform_cols=None, future_year=None):
    feature_cols = features.columns.tolist()
    if log_transform_cols:
        for col in log_transform_cols:
            if col in features.columns:
                features[col] = features[col].clip(lower=0)
                features[col] = np.log1p(features[col])
    if future_year is not None:
        features['Year'] = features.index.to_series().str.extract(r'_(\d{4})')[0].astype(int)
        train_features = features[features['Year'] < future_year]
    else:
        train_features = features.copy()
    means = train_features[feature_cols].mean()
    stds = train_features[feature_cols].std()
    stds[stds < 1e-8] = 1e-8
    features[feature_cols] = (features[feature_cols] - means) / stds
    if 'Year' in features.columns:
        features = features.drop(columns=['Year'])
    return features

features = normalize_features(features, log_transform_cols=['Traffic'], future_year=2023)

# --- Build Graph ---
G = nx.from_pandas_edgelist(edges)
valid_nodes = set(features.index) & set(labels.index)
G = G.subgraph(valid_nodes).copy()
node_list = sorted(G.nodes())
features = features.loc[node_list]
labels = labels.loc[node_list]

# --- Build Tensors ---
x = torch.tensor(features.values, dtype=torch.float32).to(DEVICE)
y_raw = labels.values.squeeze()

# Masks
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

# --- Normalize Labels on Training Set ---
y_train_vals = y_raw[train_mask.numpy()]
label_mean = y_train_vals.mean()
label_std = y_train_vals.std() if y_train_vals.std() > 1e-8 else 1e-8
y_norm = (y_raw - label_mean) / label_std
y = torch.tensor(y_norm, dtype=torch.float32).to(DEVICE)

# --- Adjacency Matrix ---
A = nx.to_numpy_array(G, nodelist=node_list)
A = torch.tensor(A, dtype=torch.float32).to(DEVICE)

'''
----
-----> TRAINING <-----
----
'''
LEARNING_RATE = 0.003
EPOCHS = 200

model = GCN(in_dim=x.size(1), out_dim=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

loss_history = []
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    out = model(x, A).squeeze()
    loss = F.mse_loss(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    preds = model(x, A).squeeze().cpu().numpy()
    preds_denorm = preds * label_std + label_mean
    true_vals = y_raw

test_mse = mean_squared_error(true_vals[test_mask.numpy()], preds_denorm[test_mask.numpy()])
print(f"Test MSE (denormalized): {test_mse:.4f}")

'''
----
-----> METRICS <-----
----
'''
fig, axes = plt.subplots(1, 3, figsize=(32, 6), gridspec_kw={'wspace': 0.2})

# Scatter
axes[0].scatter(true_vals, preds_denorm, alpha=0.7)
axes[0].plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
axes[0].set_xlabel("True Value")
axes[0].set_ylabel("Predicted Value")
axes[0].set_title("True vs Predicted Scatter")
axes[0].grid(True)

# Loss curve
axes[1].plot(loss_history)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Training Loss (MSE)")
axes[1].set_title("Training Loss Curve")
axes[1].grid(True)
axes[1].text(0.05, 0.95, f"MSE: {test_mse:.4f}",
             transform=axes[0].transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Histograms
axes[2].hist(true_vals, bins=30, alpha=0.6, label='True Labels', edgecolor='black')
axes[2].hist(preds_denorm, bins=30, alpha=0.6, label='Predicted Values', edgecolor='black')
axes[2].set_title("Label vs Prediction Distribution")
axes[2].set_xlabel("Value")
axes[2].set_ylabel("Frequency")
axes[2].legend()
axes[2].grid(True)

plt.show()
