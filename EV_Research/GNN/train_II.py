# ===== train_prep.py =====
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from Models.GCN import GCN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Transforms import transform_features, transform_labels, inverse_transform_preds

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
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data"

# ----------------- CONFIG -----------------
TRAIN_END_YEAR = 2021
VAL_YEARS      = [2022]
TEST_YEARS     = [2023]

LOG_COLS        = ["Age", "EVs", "Education", "Income", "Length", "Male",
                   "Population", "StationCount", "White"]
NO_SCALE_COLS   = ["Policy"]
REMOVE_FEATURES = ['Temperature']

LABEL_METHOD       = "zscore"   # "zscore", "asinh", "yeo-johnson", "quantile"
ASINH_SCALE        = "iqr"
STANDARDIZE_AFTER  = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST = 'tampa'
SPATIO_TEMPORAL = False

# --------- LOAD DATA ---------
if SPATIO_TEMPORAL:
    edges = pd.read_csv(data_dir / TEST / "edges_fullscale_with_time.csv")
else:
    edges = pd.read_csv(data_dir / TEST / "edges_fullscale.csv")

features = pd.read_csv(data_dir / TEST / "Final_Joined_Features.csv", index_col=0)
labels   = pd.read_csv(data_dir / TEST / "Labels_Test.csv", index_col=0)

# ---- Build Graph (ensure correct column names if different) ----
# e.g., nx.from_pandas_edgelist(edges, source="src", target="dst")
G = nx.from_pandas_edgelist(edges)  # assumes 'source','target' cols

# Align nodes shared by features and labels
valid_nodes = set(features.index) & set(labels.index)
G = G.subgraph(valid_nodes).copy()

# Stable node order; keep this for X/y row order AND edge_index mapping
node_list = sorted(G.nodes())

features = features.loc[node_list]
for f in REMOVE_FEATURES:
    if f in features.columns:
        features = features.drop(columns=f)

labels   = labels.loc[node_list]

# ----------------- HELPERS -----------------
def extract_years_from_index(idx: pd.Index) -> np.ndarray:
    # expects index like "Road123_2020"
    return idx.to_series().str.extract(r'_(\d{4})')[0].astype(int).to_numpy()

def build_edge_index_from_G(G: nx.Graph, node_list: list) -> torch.Tensor:
    """Returns PyG-style edge_index (2, E) consistent with row order of X/y."""
    node_id = {n: i for i, n in enumerate(node_list)}
    rows = []
    cols = []
    # For undirected graphs, add both (u,v) and (v,u) to match PyG convention
    for u, v in G.edges():
        if u in node_id and v in node_id:
            ui = node_id[u]; vi = node_id[v]
            rows.extend([ui, vi])
            cols.extend([vi, ui])
    ei = torch.tensor([rows, cols], dtype=torch.long)
    return ei

# ----------------- 1) Time-based masks -----------------
assert (features.index == labels.index).all(), "Features and labels index misaligned!"
years = extract_years_from_index(features.index)

train_mask = torch.from_numpy(years <= TRAIN_END_YEAR).to(torch.bool).to(DEVICE)
val_mask   = torch.from_numpy(np.isin(years, VAL_YEARS)).to(torch.bool).to(DEVICE)
test_mask  = torch.from_numpy(np.isin(years, TEST_YEARS)).to(torch.bool).to(DEVICE)

print(f"Train rows: {int(train_mask.sum())} | Val rows: {int(val_mask.sum())} | Test rows: {int(test_mask.sum())}")

# ----------------- 2) Transform FEATURES -----------------
features_tx, feat_summary = transform_features(
    features=features,
    train_mask=train_mask,     # boolean torch tensor ok if your fn handles it; otherwise pass .cpu().numpy()
    log_cols=LOG_COLS,
    no_scale_cols=NO_SCALE_COLS,
    temperature_fill=True,
    verbose=True
)

X = torch.tensor(features_tx.values, dtype=torch.float32, device=DEVICE)

# ----------------- 3) Transform LABELS -----------------
y, y_meta = transform_labels(
    labels_df=labels,
    train_mask=train_mask,
    method=LABEL_METHOD,
    asinh_scale=ASINH_SCALE,
    standardize_after=STANDARDIZE_AFTER,
    device=DEVICE
)

# ----------------- 4) Edge index for PyG -----------------
edge_index = build_edge_index_from_G(G, node_list).to(DEVICE)

# ----------------- 5) Sanity prints -----------------
# print(f"\nTensors ready: X.shape={tuple(X.shape)}, y.shape={tuple(y.shape)} on {DEVICE}")
# print(f"edge_index shape: {tuple(edge_index.shape)} (should be [2, E])")

# === PREP CHECKS: paste right after you build X, y, edge_index, and masks ===
print("X:", tuple(X.shape), "y:", tuple(y.shape), "edge_index:", tuple(edge_index.shape))
print("train/val/test:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))
print("y (tx) min/max [train]:", float(y[train_mask].min()), float(y[train_mask].max()))
print("y_meta:", y_meta)   # dict or sklearn transformer

# Assumes you already have:
#   DEVICE, X, y, edge_index, train_mask, val_mask, test_mask
#   labels  (original labels DataFrame, same row order as X/y)
#   model   (your GCN model defined elsewhere, using PyG GCNConv)
# -------- Training config --------
LEARNING_RATE = 5e-4
EPOCHS = 300
MAX_NORM = 1.0
STEP_EVERY = 50
GAMMA = 0.5
# --- Regularization / early stop ---
WEIGHT_DECAY = 5e-4          # classic for GCNs
EARLY_STOP_PATIENCE = 8      # was 30
EARLY_STOP_MIN_DELTA = 1e-3  # require a bit more improvement


model = GCN(in_dim=X.size(1), out_dim=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# replace your scheduler init:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5, verbose=False
)

train_losses, val_losses = [], []

best_state = None
best_val = float("inf")
best_epoch = -1
no_improve = 0

for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    optimizer.zero_grad()

    out = model(X, edge_index).squeeze(-1)   # [N]
    if epoch == 0:
        print("pred (tx) min/max @epoch0:", float(out.min()), float(out.max()))

    #loss = F.mse_loss(out[train_mask], y[train_mask])
    loss = F.huber_loss(out[train_mask], y[train_mask], delta=1.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
    optimizer.step()
    #scheduler.step()

    train_losses.append(loss.item())

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        val_out = model(X, edge_index).squeeze(-1)
        #val_loss = F.mse_loss(val_out[val_mask], y[val_mask]).item()
        val_loss = F.huber_loss(val_out[val_mask], y[val_mask], delta=1.0).item()
        # and in the loop, after you compute val_loss:
        scheduler.step(val_loss)
        val_losses.append(val_loss)

    # ---- early stopping bookkeeping ----
    if val_loss + EARLY_STOP_MIN_DELTA < best_val:
        best_val = val_loss
        best_epoch = epoch
        # store best weights on CPU to avoid GPU mem growth
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch} with val={best_val:.4f})")
            break

# ---- restore best weights ----
if best_state is not None:
    model.load_state_dict(best_state)

# -------- Evaluation (on test) --------
model.eval()
with torch.no_grad():
    preds = model(X, edge_index).squeeze(-1).detach().cpu().numpy()

# print("preds (tx) min/max:", float(np.nanmin(preds)), float(np.nanmax(preds)))
# Denormalize predictions back to raw label scale
preds_denorm = inverse_transform_preds(preds, y_meta)

# Raw (original) true labels from the DataFrame
true_vals = labels.select_dtypes(include=[np.number]).values.squeeze()

# ----- Baselines on RAW scale -----
test_idx  = test_mask.detach().cpu().numpy().astype(bool)
train_idx = train_mask.detach().cpu().numpy().astype(bool)

y_train_raw = true_vals[train_idx]
y_test_raw  = true_vals[test_idx]

# Train mean / median baselines
mean_baseline   = np.full_like(y_test_raw, y_train_raw.mean(), dtype=float)
median_baseline = np.full_like(y_test_raw, np.median(y_train_raw), dtype=float)

# --- Carry-forward baseline (ŷ_t = y_{t-1}) ---
# Expect index like "Road123_2020"
idx = labels.index.astype(str)

# Safest: extract final 4-digit year via regex
parts = idx.str.extract(r'^(.*)_(\d{4})$')  # parts[0]=base, parts[1]=year (str)
base = parts[0]
year = pd.to_numeric(parts[1], errors="coerce")

# Build previous-year index; keep NA where year missing
year_prev = (year - 1).astype("Int64")  # nullable int
prev_index = (base + "_" + year_prev.astype(str)).to_numpy()

# Map raw labels by index and fetch previous-year values
y_series = pd.Series(true_vals, index=idx)
cf_all = y_series.reindex(prev_index).to_numpy()

# Test-split carry-forward values, fill missing with train mean
cf_test = cf_all[test_idx].astype(float)
miss = np.isnan(cf_test)
if miss.any():
    cf_test[miss] = y_train_raw.mean()

print("\n--- Baseline metrics (raw scale, test split) ---")
print("Model MSE:",   mean_squared_error(y_test_raw, preds_denorm[test_idx]))
print("Model MAE:",   mean_absolute_error(y_test_raw,  preds_denorm[test_idx]))
print("Mean  MSE:",   mean_squared_error(y_test_raw, mean_baseline))
print("Mean  MAE:",   mean_absolute_error(y_test_raw,  mean_baseline))
print("Median MSE:",  mean_squared_error(y_test_raw, median_baseline))
print("Median MAE:",  mean_absolute_error(y_test_raw,  median_baseline))
print("CarryF MSE:",  mean_squared_error(y_test_raw, cf_test))
print("CarryF MAE:",  mean_absolute_error(y_test_raw,  cf_test))


# -------- Plots --------
fig, axes = plt.subplots(1, 3, figsize=(32, 6), gridspec_kw={'wspace': 0.2})

# Scatter: True vs Pred (denormed)
axes[0].scatter(true_vals, preds_denorm, alpha=0.7)
mn, mx = np.nanmin(true_vals), np.nanmax(true_vals)
axes[0].plot([mn, mx], [mn, mx])
axes[0].set_xlabel("True Value")
axes[0].set_ylabel("Predicted Value")
axes[0].set_title("True vs Predicted (raw scale)")
axes[0].grid(True)

# Loss curves
axes[1].plot(train_losses, label="Train Loss")
axes[1].plot(val_losses, label="Val Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].set_title("Training vs Validation Loss")
axes[1].legend()
axes[1].grid(True)

# Histograms
axes[2].hist(true_vals, bins=30, alpha=0.6, label="True")
axes[2].hist(preds_denorm, bins=30, alpha=0.6, label="Predicted")
axes[2].set_title("Label vs Prediction Distribution (raw scale)")
axes[2].set_xlabel("Value")
axes[2].set_ylabel("Frequency")
axes[2].legend()
axes[2].grid(True)

plt.show()
