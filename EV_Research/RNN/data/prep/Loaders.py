# EV_Research/RNN/data/loaders.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
except Exception as e:
    raise ImportError("Step 3 requires PyTorch. Please install torch.") from e

# --- make "from EV_Research..." work when running this file directly ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # parent of EV_Research
# ----------------------------------------------------------------------

# Step 2.5 helpers (your file is CamelCase in data/prep/)
from EV_Research.RNN.data.prep.Normalize_City import (
    load_feature_scaler,
    load_label_scaler,
    apply_feature_transform_inplace,
    transform_labels_inplace,
    _load_feature_names,
    _years_vector,
)

def _augment_with_lag_and_time(
    X_trf: np.ndarray,          # [T,N,F_kept] (already feature-transformed if apply_in_loader=True)
    years_vec: np.ndarray,      # [T] sorted by t_idx (e.g., 2016..2023)
    y_raw: np.ndarray,          # [T,N] raw labels BEFORE transform (you already load these)
    y_mask: np.ndarray,         # [T,N] bool
    labels_dir: Path,           # data/<city>/labels_RNN
    cfg: dict,                  # full YAML dict
) -> tuple[np.ndarray, list[str]]:
    """Return (X_aug, extra_feature_names). Adds lag1_label_z, lag1_available, and t_norm if enabled."""
    ds_cfg = cfg.get("dataset", {})
    add_lag    = bool(ds_cfg.get("use_lag1_label_feature", False))
    add_lag_av = bool(ds_cfg.get("lag1_add_availability_flag", False))
    add_time   = bool(ds_cfg.get("add_time_feature", False))

    extras = []
    names  = []

    # --- 1) lag1_label_z (+ optional availability flag) ---
    if add_lag or add_lag_av:
        # load label scaler and go to z-space
        ls = load_label_scaler(labels_dir)
        y_z = transform_labels_inplace(y_raw.astype(np.float32, copy=False), ls)  # [T,N]

        T, N = y_z.shape
        lag1 = np.zeros_like(y_z, dtype=np.float32)       # will be zeros at t=0 (=> mean in z)
        lag_ok = np.zeros_like(y_mask, dtype=bool)

        # shift by one in time
        lag1[1:]  = y_z[:-1]
        lag_ok[1:] = y_mask[:-1]

        # zero-out masked positions if you zero masked features elsewhere
        if bool(ds_cfg.get("zero_masked_features", True)):
            lag1[~lag_ok] = 0.0   # 0 == mean in z-space

        if add_lag:
            extras.append(lag1[:, :, None])         # [T,N,1]
            names.append("lag1_label_z")

        if add_lag_av:
            extras.append(lag_ok.astype(np.float32)[:, :, None])   # [T,N,1]
            names.append("lag1_available")

    # --- 2) t_norm ∈ [0,1] (frozen using train range) ---
    if add_time:
        norm_cfg = cfg.get("normalize", {})
        train_end_year  = int(norm_cfg.get("train_end_year"))
        min_train_year  = int(min(y for y in years_vec if y <= train_end_year))
        denom = max(1, train_end_year - min_train_year)
        t_norm = (years_vec.astype(np.float32) - min_train_year) / float(denom)
        t_norm = np.clip(t_norm, 0.0, 1.0)  # cap val/test to 1.0
        # broadcast to [T,N,1]
        t_b = np.broadcast_to(t_norm[:, None, None], (X_trf.shape[0], X_trf.shape[1], 1)).copy()
        extras.append(t_b)
        names.append("t_norm")

    if not extras:
        return X_trf, []

    X_aug = np.concatenate([X_trf, *extras], axis=-1)   # [T,N,F+K]
    return X_aug, names

# Step 2 adjacency helper (fallback to compute norms if needed)
def _norm_rw(A: np.ndarray, add_self_loops: bool) -> np.ndarray:
    if add_self_loops:
        A = A.copy()
        np.fill_diagonal(A, A.diagonal() + 1.0)
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return (A / rowsum).astype(np.float32)

def _norm_gcn(A: np.ndarray, add_self_loops: bool) -> np.ndarray:
    if add_self_loops:
        A = A.copy()
        np.fill_diagonal(A, A.diagonal() + 1.0)
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    Dm12 = np.diag(1.0 / np.sqrt(d))
    return (Dm12 @ A @ Dm12).astype(np.float32)

def _load_yaml(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _find_project_root_from_cfg(cfg_path: Path) -> Path:
    """
    Walk upward until we find the outer project root that has:
      - a sibling 'data' directory
      - EV_Research/RNN/config/default.yaml
    """
    p = cfg_path.resolve()
    for _ in range(8):
        cand = p.parent
        if (cand / "data").exists() and (cand / "EV_Research" / "RNN" / "config" / "default.yaml").exists():
            return cand
        p = cand
    # fallback to three parents above cfg (your known layout)
    return cfg_path.resolve().parents[3]

def _resolve_city_paths(project_root: Path, city: str) -> Dict[str, Path]:
    root = project_root / "data" / city
    return {
        "city_root": root,
        "features_dir": root / "features_RNN",
        "labels_dir": root / "labels_RNN",
        "graph_dir": root / "graph_static",
    }

def _load_adjacency(graph_dir: Path, norm: str, add_self_loops: bool) -> np.ndarray:
    """
    Load a normalized adjacency if present. Be robust to:
      - numpy savez (arr_0)
      - numpy savez with named key (e.g., 'A')
      - scipy.sparse.save_npz format (data/indices/indptr/shape)
    Fallback: compute from A_raw using the requested norm.
    """
    import numpy as _np

    # 1) Try common normalized files first
    candidates = []
    if norm == "rw":
        candidates = ["A_rw.npz", "A_rw_sl1.npz"]
    elif norm == "gcn":
        candidates = ["A_gcn.npz", "A_gcn_sl1.npz"]
    else:
        raise ValueError("graph.norm must be 'rw' or 'gcn'")

    # Always consider raw as fallback
    candidates += ["A_raw.npz", "A.npz", "adj.npz"]

    def _load_any_npz(path: Path) -> Optional[np.ndarray]:
        """Return dense array if this npz is loadable, else None."""
        # Try SciPy sparse first (quietly)
        try:
            from scipy.sparse import load_npz as _sp_load_npz, issparse as _sp_issparse
            try:
                S = _sp_load_npz(path)
                if _sp_issparse(S):
                    return S.toarray().astype(np.float32)
            except Exception:
                pass
        except Exception:
            # SciPy not installed or not a SciPy .npz; continue
            pass

        # Try numpy npz keys
        try:
            with _np.load(path, allow_pickle=False) as z:
                keys = list(z.keys())
                if "arr_0" in z:
                    return z["arr_0"].astype(np.float32)
                # common named keys
                for k in ("A", "adjacency", "adj", "matrix"):
                    if k in z:
                        return z[k].astype(np.float32)
                # SciPy-style keys (manual reconstruct)
                if set(keys) >= {"data", "indices", "indptr", "shape"}:
                    from scipy.sparse import csr_matrix as _csr
                    S = _csr((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
                    return S.toarray().astype(np.float32)
                # Unknown structure: print keys to help debug
                raise KeyError(f"Unrecognized keys in {path.name}: {keys}")
        except Exception:
            return None

    # Try candidates in order
    for fname in candidates:
        p = graph_dir / fname
        if p.exists():
            A = _load_any_npz(p)
            if A is not None:
                # If we loaded a normalized file that already has self-loops, we don't add again.
                if fname.startswith("A_raw"):
                    # Normalize from raw
                    if norm == "rw":
                        return _norm_rw(A, add_self_loops)
                    else:
                        return _norm_gcn(A, add_self_loops)
                else:
                    # Assume file already matches requested norm; optionally add self-loops if name suggests no SL
                    if add_self_loops and "sl1" not in fname.lower():
                        A = A.copy()
                        np.fill_diagonal(A, A.diagonal() + 1.0)
                    return A.astype(np.float32)

    # If nothing worked, explicit fallback: require A_raw and normalize
    A_raw_path = graph_dir / "A_raw.npz"
    if not A_raw_path.exists():
        raise FileNotFoundError(
            f"Missing adjacency in {graph_dir}. Tried {candidates}. "
            f"Please ensure A_raw.npz exists."
        )
    # Last-resort raw load (robust again)
    A_raw = _load_any_npz(A_raw_path)
    if A_raw is None:
        raise RuntimeError(f"Could not read {A_raw_path} (unknown npz format).")
    return _norm_rw(A_raw, add_self_loops) if norm == "rw" else _norm_gcn(A_raw, add_self_loops)

class EVSequenceDataset(Dataset):
    """
    Builds rolling windows:
      input:  [t-L .. t-1]
      target: y[t]  (already Δ(t->t+2) at row t)
    Keeps sample if window fits and y_mask[t].any()==True.
    """
    def __init__(self, cfg_path: Path):
        self.cfg = _load_yaml(cfg_path)
        self.project_root = _find_project_root_from_cfg(cfg_path)

        city = self.cfg["dataset"]["city"]
        paths = _resolve_city_paths(self.project_root, city)

        # Load arrays
        X = np.load(paths["features_dir"] / "X.npy", mmap_mode="r")         # [T,N,F]
        mask = np.load(paths["features_dir"] / "mask.npy", mmap_mode="r")   # [T,N]
        y = np.load(paths["labels_dir"] / "y.npy", mmap_mode="r")           # [T,N]
        y_mask = np.load(paths["labels_dir"] / "y_mask.npy", mmap_mode="r") # [T,N]
        years_vec, _ = _years_vector(paths["features_dir"])                 # [T]
        feat_names = _load_feature_names(paths["features_dir"])

        # Load scalers
        fsc = load_feature_scaler(paths["features_dir"])
        lsc = load_label_scaler(paths["labels_dir"])

        # Apply transforms
        X_trf, kept = apply_feature_transform_inplace(X, feat_names, fsc)   # [T,N,F_kept]
        y_trf = transform_labels_inplace(y, lsc)                             # [T,N]

        # === NEW: add lag1_label_z / lag1_available / t_norm ===
        X_trf, extra_names = _augment_with_lag_and_time(
            X_trf=X_trf,
            years_vec=years_vec,
            y_raw=y,                     # RAW labels (pre-transform)
            y_mask=y_mask,
            labels_dir=paths["labels_dir"],
            cfg=self.cfg,
        )
        if extra_names:
            kept = list(kept) + extra_names

        # Optionally zero masked features
        zero_masked = bool(self.cfg.get("dataset", {}).get("zero_masked_features", True))
        if zero_masked:
            # mask: [T,N] → [T,N,1] for broadcast
            X_trf = X_trf * mask[..., None].astype(np.float32)

        # Save transformed arrays (kept only in memory in this dataset)
        self.X_trf = X_trf.astype(np.float32)
        self.mask = mask.astype(bool)
        self.y_trf = y_trf.astype(np.float32)
        self.y_mask = y_mask.astype(bool)
        self.years_vec = years_vec.astype(int)
        self.kept_feature_names = kept

        # Adjacency (one copy)
        gcfg = self.cfg.get("graph", {})
        self.A = _load_adjacency(paths["graph_dir"], norm=gcfg.get("norm", "rw"), add_self_loops=bool(gcfg.get("add_self_loops", True)))

        # Build index of valid target t
        L = int(self.cfg["dataset"].get("window_L", 4))
        stride = int(self.cfg["dataset"].get("stride", 1))
        T = self.X_trf.shape[0]
        valid_targets = []
        dropped_years = []
        for t in range(L, T, stride):
            if not self.y_mask[t].any():
                dropped_years.append(int(self.years_vec[t]))
                continue
            valid_targets.append(t)

        self.targets_idx = np.array(valid_targets, dtype=int)  # indices t
        self.target_years = self.years_vec[self.targets_idx]   # [B]
        self.dropped_years = sorted(set(dropped_years))

        # Shapes
        self.T, self.N, self.F_kept = self.X_trf.shape

    def __len__(self):
        return len(self.targets_idx)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t = int(self.targets_idx[i])
        L = int(self.cfg["dataset"].get("window_L", 4))

        X_seq = self.X_trf[t - L:t]           # [L,N,F]
        mask_seq = self.mask[t - L:t]         # [L,N]
        y_vec = self.y_trf[t]                 # [N]
        y_mask_vec = self.y_mask[t]           # [N]
        year_t = int(self.years_vec[t])

        # Convert to tensors
        X_seq = torch.from_numpy(X_seq)               # float32
        mask_seq = torch.from_numpy(mask_seq.astype(np.float32))
        y_vec = torch.from_numpy(y_vec)
        y_mask_vec = torch.from_numpy(y_mask_vec.astype(np.float32))
        A = torch.from_numpy(self.A)                  # [N,N] float32

        # Node index (0..N-1)
        node_index = torch.arange(self.N, dtype=torch.long)

        return dict(
            X_seq=X_seq,               # [L,N,F]
            mask_seq=mask_seq,         # [L,N]
            y_trf=y_vec,               # [N]
            y_mask=y_mask_vec,         # [N]
            A=A,                       # [N,N]
            year=torch.tensor(year_t, dtype=torch.int32),
            node_index=node_index,     # [N]
            sample_idx=torch.tensor(i, dtype=torch.int64),
        )

def make_dataloader(
    cfg_path: Path,
    split: str,
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    drop_last: bool = False,
) -> Tuple[DataLoader, EVSequenceDataset, Dict[str, np.ndarray]]:
    """
    Create a DataLoader for a given split ('train' | 'val' | 'test').
    """
    from EV_Research.RNN.data.prep.Splits import build_split_indices

    ds = EVSequenceDataset(cfg_path)
    train_end_year = int(_load_yaml(cfg_path)["normalize"]["train_end_year"])
    idx = build_split_indices(ds.target_years, train_end_year)

    if split not in idx:
        raise ValueError("split must be 'train' | 'val' | 'test'")
    subset_idx = idx[split]

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        Subset(ds, subset_idx.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return loader, ds, idx

def check_sequences(cfg_path: Path, examples: int = 3) -> None:
    """
    Diagnostics: prints shapes, example windows, dropped target years, coverage, adjacency info, and basic label stats.
    """
    cfg = _load_yaml(cfg_path)
    ds = EVSequenceDataset(cfg_path)

    print("\n[Step 3] Sequence Builder Check")
    print(f"  City:                 {cfg['dataset']['city']}")
    print(f"  Window L:             {cfg['dataset'].get('window_L', 4)}")
    print(f"  Stride:               {cfg['dataset'].get('stride', 1)}")
    print(f"  Samples (kept):       {len(ds)}")
    print(f"  Dropped target years: {ds.dropped_years if ds.dropped_years else 'None'}")
    print(f"  Shapes: X_trf={ds.X_trf.shape}, y_trf={ds.y_trf.shape}, A={ds.A.shape}")
    print(f"  Kept features (F_kept): {ds.F_kept}")
    print(f"  First 10 feature names: {ds.kept_feature_names[:10]}")

    # Example windows
    print("\n  Example windows:")
    for i in np.linspace(0, max(len(ds)-1, 0), num=min(examples, len(ds)), dtype=int):
        t = int(ds.targets_idx[i])
        L = int(cfg["dataset"].get("window_L", 4))
        yrs_in = ds.years_vec[t - L:t].tolist()
        yr_out = int(ds.years_vec[t])
        print(f"    [{', '.join(map(str, yrs_in))}]  ->  target {yr_out}")

    # Coverage per year (mask and y_mask)
    print("\n  Coverage (% True) per year:")
    by_year = {}
    for ti, yr in enumerate(ds.years_vec):
        f_cov = float(ds.mask[ti].mean()) * 100.0
        y_cov = float(ds.y_mask[ti].mean()) * 100.0
        by_year[int(yr)] = (f_cov, y_cov)
    for yr in sorted(by_year):
        f_cov, y_cov = by_year[yr]
        print(f"    {yr}: features={f_cov:5.1f}%  labels={y_cov:5.1f}%")

    # Adjacency info
    print(f"\n  Adjacency norm:       {cfg.get('graph', {}).get('norm', 'rw')}")
    print(f"  Add self loops:       {cfg.get('graph', {}).get('add_self_loops', True)}")

    # Train label stats (quick sanity)
    from EV_Research.RNN.data.prep.Splits import build_split_masks

    masks = build_split_masks(ds.target_years, int(cfg["normalize"]["train_end_year"]))
    if masks["train"].any():
        y_train = ds.y_trf[ds.targets_idx[masks["train"]]]  # [Btrain, N]
        m_train = ds.y_mask[ds.targets_idx[masks["train"]]] # [Btrain, N]
        vals = y_train[m_train]
        if vals.size > 0:
            mean = float(np.nanmean(vals))
            std  = float(np.nanstd(vals, ddof=0))
            print(f"\n  Train targets (transformed) mean≈{mean:+.4f}  std≈{std:.4f}")
        else:
            print("\n  Train targets: no valid (masked) label entries found.")
    else:
        print("\n  Train split has no samples with current train_end_year.")
