# ====== RNN GCN TRAINING: GCN -> GRU (+ optional attention) ======
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
from tqdm import tqdm

# -------------------- CONFIG --------------------
DATA_DIR = Path(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa")

FEATURES_DIR = DATA_DIR / "features_RNN"
LABELS_DIR   = DATA_DIR / "labels_RNN"
EDGES_DIR    = DATA_DIR / "edges_RNN_synced"   # <-- use the *synced* folder

USE_ATTENTION   = False
TASK            = "regression"  # "regression" or "classification"
HORIZON_YEARS   = 0             # your labels are growth over +2 years
TRAIN_END_YEAR  = 2021          # train up to this (inclusive)
VAL_YEARS       = [2022]        # validate on these
TEST_YEARS      = [2023]        # test on these

EPOCHS          = 200
LR              = 0.007
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.3
GCN_DIM         = 128
RNN_DIM         = 128
CLIP_NORM       = 2.0
SEED            = 126
# ------------------------------------------------

# reproducibility-ish
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", DEVICE)

# --------------- IO: load tensors & metadata ---------------
X_np   = np.load(FEATURES_DIR / "X.npy")        # (T,N,F)
M_np   = np.load(FEATURES_DIR / "mask.npy")     # (T,N) bool
y_np   = np.load(LABELS_DIR / "y.npy")          # (T,N)
Ym_np  = np.load(LABELS_DIR / "y_mask.npy")     # (T,N) bool

years_df = pd.read_csv(FEATURES_DIR / "years.csv").sort_values("t_idx")
years = years_df["year"].to_list()
roads_df = pd.read_csv(FEATURES_DIR / "road_ids.csv").sort_values("node_id")
roads = roads_df["road_id"].to_list()

assert X_np.shape[0] == len(years), "T mismatch between X and years.csv"
assert X_np.shape[1] == len(roads), "N mismatch between X and road_ids.csv"
assert X_np.shape[:2] == y_np.shape[:2] == M_np.shape == Ym_np.shape, "Shape mismatch"

T, N, F_in = X_np.shape
print(f"Loaded: X(T,N,F)={X_np.shape}, y(T,N)={y_np.shape}, years={years}")

# torch tensors
X  = torch.from_numpy(X_np).float().to(DEVICE)
M  = torch.from_numpy(M_np).bool().to(DEVICE)
y  = torch.from_numpy(y_np).float().to(DEVICE)
Ym = torch.from_numpy(Ym_np).bool().to(DEVICE)

# --------------- Edges -> normalized sparse A_hat ---------------
def build_normalized_adj(edge_index: torch.Tensor, num_nodes: int, device=None):
    """
    edge_index: (2,E) LongTensor (can be undirected or directed; we'll symmetrize)
    returns A_hat = D^{-1/2} (A+I) D^{-1/2} as torch.sparse.FloatTensor (N,N)
    """
    if device is None:
        device = edge_index.device
    N = num_nodes

    # add self loops
    self_loops = torch.arange(N, device=device)
    ei = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)

    # force undirected
    ei_rev = ei.flip(0)
    ei = torch.cat([ei, ei_rev], dim=1)

    vals = torch.ones(ei.shape[1], device=device)
    A = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = (deg + 1e-12).pow(-0.5)

    row, col = A.indices()
    norm_vals = deg_inv_sqrt[row] * A.values() * deg_inv_sqrt[col]
    A_hat = torch.sparse_coo_tensor(A.indices(), norm_vals, (N, N)).coalesce()
    return A_hat

edges_idx = pd.read_csv(EDGES_DIR / "edges_static_indexed.csv")
edge_index = torch.tensor(edges_idx[['source_id','target_id']].values.T, dtype=torch.long, device=DEVICE)
A_hat = build_normalized_adj(edge_index, num_nodes=N, device=DEVICE)
print("Adjacency ready.")

# ---------------- Models: GCN, Attention, Temporal ----------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, X, A_sparse):
        # X: (N,F), A_sparse: (N,N)
        h = torch.sparse.mm(A_sparse, X)
        return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim=128, dropout=0.3):
        super().__init__()
        self.g1 = GCNLayer(in_dim, 128)
        self.g2 = GCNLayer(128, out_dim)
        self.dropout = dropout
    def forward(self, X, A):
        h = self.g1(X, A);  h = F.relu(h);  h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(h, A);  h = F.relu(h)
        return h  # (N, out_dim)

class TemporalAttention(nn.Module):
    """
    Simple additive attention over time, per node.
    Given H: (T,N,D), returns context C: (T,N,D) where C_t attends over H_{<=t}.
    """
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1, bias=False)
    def forward(self, H):
        T, N, D = H.shape
        Q = torch.tanh(self.fc(H))        # (T,N,D)
        e = self.score(Q).squeeze(-1)     # (T,N)
        alphas = []
        for t in range(T):
            a = torch.softmax(e[:t+1], dim=0)   # (t+1,N)
            pad = torch.zeros(T-(t+1), N, device=H.device)
            alphas.append(torch.vstack([a, pad]))
        A = torch.stack(alphas, dim=0)    # (T,T,N)
        # C_t = sum_{s<=t} A[t,s,n] * H_s
        H_exp = H.permute(1,0,2)          # (N,T,D)
        A_exp = A.permute(2,0,1)          # (N,T,T)
        C = torch.bmm(A_exp, H_exp)       # (N,T,D)
        return C.permute(1,0,2)           # (T,N,D)

class GCN_GRU_Temporal(nn.Module):
    def __init__(self, in_dim, gcn_dim=128, rnn_dim=128, out_dim=1, use_attention=True, dropout=0.3):
        super().__init__()
        self.gcn = GCN(in_dim, out_dim=gcn_dim, dropout=dropout)
        self.gru = nn.GRU(input_size=gcn_dim, hidden_size=rnn_dim, batch_first=False)
        self.use_attention = use_attention
        if use_attention:
            self.attn = TemporalAttention(rnn_dim)
        self.head = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, out_dim)
        )
    def forward(self, X, A_sparse):
        # X: (T,N,F)
        T, N, F = X.shape
        Rt = [self.gcn(X[t], A_sparse) for t in range(T)]  # list of (N,gcn_dim)
        R  = torch.stack(Rt, dim=0)                        # (T,N,gcn_dim)
        H, _ = self.gru(R)                                 # (T,N,rnn_dim)
        Z = self.attn(H) if self.use_attention else H      # (T,N,rnn_dim)
        Y = self.head(Z)                                   # (T,N,out_dim)
        return Y.squeeze(-1)                               # -> (T,N) if out_dim=1

# ---------------------- Losses ----------------------
def masked_mse(pred, y, mask):
    diff = (pred - y)**2
    diff = diff[mask]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return diff.mean()

def masked_bce_with_logits(pred, y, mask, pos_weight=None):
    # y expected in {0,1}, pred are logits
    loss = F.binary_cross_entropy_with_logits(pred[mask], y[mask], pos_weight=pos_weight)
    return loss

# ----------------- Time splits & masks -----------------
# Build time indices respecting horizon: no labels exist for the last HORIZON_YEARS
def time_indices(years: List[int], train_end: int, val_years: List[int], test_years: List[int], horizon: int):
    t_train = [i for i,y in enumerate(years) if y <= train_end]
    t_val   = [i for i,y in enumerate(years) if y in val_years]
    t_test  = [i for i,y in enumerate(years) if y in test_years]
    T_last = len(years) - 1
    max_t_with_label = T_last - horizon
    filt = lambda arr: [t for t in arr if t <= max_t_with_label]
    return filt(t_train), filt(t_val), filt(t_test)

t_train, t_val, t_test = time_indices(years, TRAIN_END_YEAR, VAL_YEARS, TEST_YEARS, HORIZON_YEARS)
print("Train years:", [years[t] for t in t_train])
print("Val years  :", [years[t] for t in t_val])
print("Test years :", [years[t] for t in t_test])

def time_mask(base_mask: torch.Tensor, t_idx: List[int]):
    Tmask = torch.zeros_like(base_mask, dtype=torch.bool)
    if len(t_idx):
        Tmask[t_idx, :] = True
    return base_mask & Tmask

# -------------------- Model / Optim --------------------
out_dim = 1 if TASK == "regression" else 1
model = GCN_GRU_Temporal(in_dim=F_in, gcn_dim=GCN_DIM, rnn_dim=RNN_DIM,
                         out_dim=out_dim, use_attention=USE_ATTENTION, dropout=DROPOUT).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

print(sum(p.numel() for p in model.parameters()), "parameters")

# -------------------- Training loop --------------------
train_losses = []
val_losses   = []

best_val, bad, patience = math.inf, 0, 20
for epoch in tqdm(range(EPOCHS), desc="Training..."):
    # ---- train ----
    model.train()
    opt.zero_grad()
    y_hat = model(X, A_hat)  # (T,N)

    if TASK == "regression":
        loss_tr = masked_mse(y_hat, y, time_mask(Ym, t_train))
    else:
        loss_tr = masked_bce_with_logits(y_hat, y, time_mask(Ym, t_train))

    loss_tr.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    opt.step()

    # ---- val ----
    model.eval()
    with torch.no_grad():
        y_hat = model(X, A_hat)
        if TASK == "regression":
            loss_val = masked_mse(y_hat, y, time_mask(Ym, t_val))
        else:
            loss_val = masked_bce_with_logits(y_hat, y, time_mask(Ym, t_val))

    sched.step(loss_val)

    # Store for curve
    train_losses.append(loss_tr.item())
    val_losses.append(loss_val.item())

    # early stop
    # track best but no stopping
    if loss_val.item() < best_val - 1e-6:
        best_val = loss_val.item()
        torch.save(model.state_dict(), DATA_DIR / "best_temporal.pt")


# -------------------- Plot loss curves --------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Test eval --------------------
model.load_state_dict(torch.load(DATA_DIR / "best_temporal.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    y_hat = model(X, A_hat)

if TASK == "regression":
    test_loss = masked_mse(y_hat, y, time_mask(Ym, t_test)).item()
    print("Test MSE:", test_loss)
else:
    # Example: compute accuracy on test timesteps
    prob = torch.sigmoid(y_hat)
    pred_bin = (prob >= 0.5).float()
    mask = time_mask(Ym, t_test)
    correct = (pred_bin[mask] == y[mask]).float().mean().item()
    print("Test Acc:", correct)

# -------------------- Save predictions per year (optional) --------------------
out_pred = DATA_DIR / "predictions_per_year.csv"
with torch.no_grad():
    YH = y_hat.detach().cpu().numpy()  # (T,N)
df = []
for t_idx, year in enumerate(years):
    row = pd.DataFrame({
        "year": year,
        "road_id": roads,
        "pred": YH[t_idx, :]
    })
    df.append(row)
pd.concat(df, axis=0).to_csv(out_pred, index=False)
print("Wrote:", out_pred)
