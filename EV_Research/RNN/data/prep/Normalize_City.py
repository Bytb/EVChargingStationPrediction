# RNN/prep/normalize_city.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import ast

import numpy as np
import pandas as pd
import yaml


# ---------------------------- Data classes ----------------------------

@dataclass
class FeatureTransformSpec:
    name: str
    kind: str  # "zscore" | "log1p" | "passthrough" | "log1p_then_zscore"
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass
class FeatureScaler:
    train_years: List[int]
    std_floor: float
    zscore_after_log1p: bool
    drop_features: List[str]
    passthrough_features: List[str]
    log1p_features: List[str]
    keep_feature_names: List[str]         # in the exact order after dropping
    transforms: List[FeatureTransformSpec]  # aligned to keep_feature_names

    def to_json(self) -> dict:
        return {
            "train_years": self.train_years,
            "std_floor": self.std_floor,
            "zscore_after_log1p": self.zscore_after_log1p,
            "drop_features": self.drop_features,
            "passthrough_features": self.passthrough_features,
            "log1p_features": self.log1p_features,
            "keep_feature_names": self.keep_feature_names,
            "transforms": [asdict(t) for t in self.transforms],
        }


@dataclass
class LabelScaler:
    method: str  # "zscore" | "asinh_then_zscore"
    train_years: List[int]
    std_floor: float
    mean: Optional[float] = None  # for zscore or the post-asinh zscore mean
    std: Optional[float] = None   # for zscore or the post-asinh zscore std
    asinh_scale: Optional[float] = None  # s in asinh(y/s) if method=asinh_then_zscore

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------- Helpers ----------------------------

def _resolve_paths_from_yaml(root: Path, city: str) -> Dict[str, Path]:
    city_root = root / "data" / city
    paths = {
        "city_root": city_root,
        "features_dir": city_root / "features_RNN",
        "labels_dir": city_root / "labels_RNN",
        "edges_dir": city_root / "edges_RNN_synced",  # not used here
        "graph_dir": city_root / "graph_static",      # not used here
    }
    return paths


def _load_yaml(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _load_feature_names(features_dir: Path) -> List[str]:
    """
    Load feature names from either feature_names.json (list of strings)
    or feature_names.txt (one name per line, or comma-separated, or a Python-list string).
    """
    cand_json = features_dir / "feature_names.json"
    cand_txt  = features_dir / "feature_names.txt"

    if cand_json.exists():
        with open(cand_json, "r", encoding="utf-8") as f:
            names = json.load(f)
    elif cand_txt.exists():
        raw = cand_txt.read_text(encoding="utf-8").strip()
        names = None
        # 1) try newline-separated
        if "\n" in raw:
            names = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # 2) else try comma-separated
        if names is None and "," in raw:
            names = [tok.strip() for tok in raw.split(",") if tok.strip()]
        # 3) else try Python literal list
        if names is None:
            try:
                lit = ast.literal_eval(raw)
                if isinstance(lit, list) and all(isinstance(s, str) for s in lit):
                    names = lit
            except Exception:
                pass
        # 4) else treat as a single name (edge case)
        if names is None:
            names = [raw] if raw else []
    else:
        raise FileNotFoundError(
            f"Missing feature names file. Expected {cand_json} or {cand_txt}"
        )

    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError("feature names must resolve to a list of strings.")
    return names


def _years_vector(features_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (years_sorted, t_idx_sorted)."""
    years_csv = features_dir / "years.csv"
    if not years_csv.exists():
        raise FileNotFoundError(f"Missing years.csv at {years_csv}")
    dfy = pd.read_csv(years_csv)
    # Expect columns: 'year', 't_idx'
    if "year" not in dfy.columns or "t_idx" not in dfy.columns:
        raise ValueError("years.csv must contain 'year' and 't_idx' columns.")
    dfy = dfy.sort_values("t_idx")
    return dfy["year"].to_numpy(), dfy["t_idx"].to_numpy()


def _train_years(years: np.ndarray, train_end_year: int) -> List[int]:
    y_min = int(years.min())
    y_max = int(train_end_year)
    return [int(y) for y in years[(years >= y_min) & (years <= y_max)]]


def _validate_feature_lists(feature_names: List[str],
                            log1p_features: List[str],
                            passthrough_features: List[str],
                            drop_features: List[str]) -> None:
    # All names exist:
    all_lists = {
        "log1p_features": log1p_features,
        "passthrough_features": passthrough_features,
        "drop_features": drop_features,
    }
    for list_name, lst in all_lists.items():
        missing = [n for n in lst if n not in feature_names]
        if missing:
            raise ValueError(f"{list_name} contains names not in feature_names: {missing}")

    # No overlaps:
    overlaps = set(log1p_features) & set(passthrough_features) \
               | set(log1p_features) & set(drop_features) \
               | set(passthrough_features) & set(drop_features)
    if overlaps:
        raise ValueError(f"Feature lists must be disjoint; overlaps found: {sorted(overlaps)}")


def _index_map(feature_names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(feature_names)}


def _safe_std(x: np.ndarray, floor: float) -> float:
    s = float(np.nanstd(x, ddof=0))
    return max(s, floor)


# ---------------------------- Fitting ----------------------------

def fit_feature_scaler(
    X: np.ndarray,           # [T,N,F]
    mask: np.ndarray,        # [T,N] bool
    years_vec: np.ndarray,   # [T] sorted by t_idx
    feature_names: List[str],
    train_end_year: int,
    std_floor: float,
    log1p_features: List[str],
    passthrough_features: List[str],
    drop_features: List[str],
    zscore_after_log1p: bool = False,
) -> FeatureScaler:
    """
    Compute per-feature transforms using only the train slice (years <= train_end_year).
    """
    T, N, F = X.shape
    assert mask.shape == (T, N)
    assert len(feature_names) == F

    _validate_feature_lists(feature_names, log1p_features, passthrough_features, drop_features)
    idx_map = _index_map(feature_names)

    # Train slice mask
    train_years = _train_years(years_vec, train_end_year)
    train_years_set = set(train_years)
    t_mask = np.array([y in train_years_set for y in years_vec], dtype=bool)  # [T]
    valid = (mask & t_mask[:, None])  # [T,N]

    # Fail-fast: log1p features must be nonnegative everywhere (entire X, not just train)
    neg_violations = []
    for name in log1p_features:
        col = X[:, :, idx_map[name]]
        if np.nanmin(col) < 0:
            neg_violations.append(name)
    if neg_violations:
        raise ValueError(f"log1p_features contain negatives: {neg_violations}. "
                         f"Please remove/shift these or move them to z-score or passthrough.")

    # Build the kept feature order (drop excluded)
    keep_feature_names = [n for n in feature_names if n not in set(drop_features)]
    transforms: List[FeatureTransformSpec] = []

    # Compute stats for zscored (and optionally log1p_then_zscore) using ONLY the train slice where mask True
    for name in keep_feature_names:
        j = idx_map[name]
        kind = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if name in passthrough_features:
            kind = "passthrough"

        elif name in log1p_features:
            if zscore_after_log1p:
                # z-score on log1p-transformed values using train slice
                col = X[:, :, j]
                col_log = np.log1p(col)
                vals = col_log[valid]  # vector of train values
                mean_val = float(np.nanmean(vals)) if vals.size > 0 else 0.0
                std_val = _safe_std(vals, std_floor) if vals.size > 0 else 1.0
                kind = "log1p_then_zscore"
            else:
                kind = "log1p"  # no mean/std

        else:
            # default z-score
            col = X[:, :, j]
            vals = col[valid]
            mean_val = float(np.nanmean(vals)) if vals.size > 0 else 0.0
            std_val = _safe_std(vals, std_floor) if vals.size > 0 else 1.0
            kind = "zscore"

        transforms.append(FeatureTransformSpec(name=name, kind=kind, mean=mean_val, std=std_val))

    return FeatureScaler(
        train_years=train_years,
        std_floor=std_floor,
        zscore_after_log1p=zscore_after_log1p,
        drop_features=drop_features,
        passthrough_features=passthrough_features,
        log1p_features=log1p_features,
        keep_feature_names=keep_feature_names,
        transforms=transforms,
    )


def fit_label_scaler(
    y: np.ndarray,            # [T,N]
    y_mask: np.ndarray,       # [T,N] bool
    years_vec: np.ndarray,    # [T]
    train_end_year: int,
    std_floor: float,
    method: str = "zscore",   # "zscore" | "asinh_then_zscore"
    asinh_scale: Optional[float] = None,  # if None or "auto": use train std
) -> LabelScaler:
    train_years = _train_years(years_vec, train_end_year)
    t_mask = np.isin(years_vec, train_years)
    valid = (y_mask & t_mask[:, None])
    vals = y[valid]  # vector of train targets

    if method == "zscore":
        mu = float(np.nanmean(vals)) if vals.size > 0 else 0.0
        sd = _safe_std(vals, std_floor) if vals.size > 0 else 1.0
        return LabelScaler(method="zscore", train_years=train_years, std_floor=std_floor,
                           mean=mu, std=sd, asinh_scale=None)

    elif method == "asinh_then_zscore":
        # set scale s
        if asinh_scale is None or (isinstance(asinh_scale, str) and asinh_scale.lower() == "auto"):
            # train std as s (fallback to 1.0 if tiny)
            base_sd = _safe_std(vals, std_floor) if vals.size > 0 else 1.0
            s = float(base_sd)
        else:
            s = float(asinh_scale)

        vals_tr = np.arcsinh(vals / s)
        mu = float(np.nanmean(vals_tr)) if vals_tr.size > 0 else 0.0
        sd = _safe_std(vals_tr, std_floor) if vals_tr.size > 0 else 1.0
        return LabelScaler(method="asinh_then_zscore", train_years=train_years, std_floor=std_floor,
                           mean=mu, std=sd, asinh_scale=s)
    else:
        raise ValueError("normalize.method_y must be 'zscore' or 'asinh_then_zscore'.")


# ---------------------------- Apply (in-memory) ----------------------------

def apply_feature_transform_inplace(
    X: np.ndarray,
    feature_names_original: List[str],
    scaler: FeatureScaler
) -> Tuple[np.ndarray, List[str]]:
    T, N, F = X.shape
    orig_idx = _index_map(feature_names_original)
    kept = scaler.keep_feature_names
    X_out = np.empty((T, N, len(kept)), dtype=np.float32)

    for out_j, spec in enumerate(scaler.transforms):
        j = orig_idx[spec.name]
        col = X[:, :, j].astype(np.float32, copy=False)

        if spec.kind == "passthrough":
            X_out[:, :, out_j] = col

        elif spec.kind == "log1p":
            X_out[:, :, out_j] = np.log1p(col)

        elif spec.kind == "log1p_then_zscore":
            col_log = np.log1p(col)
            X_out[:, :, out_j] = (col_log - spec.mean) / (spec.std if spec.std else 1.0)

        elif spec.kind == "zscore":
            X_out[:, :, out_j] = (col - spec.mean) / (spec.std if spec.std else 1.0)

        else:
            raise ValueError(f"Unknown feature transform kind: {spec.kind}")

    return X_out, kept

# ---- Label Apply and Inverse ---- #
def transform_labels_inplace(y: np.ndarray, scaler: LabelScaler) -> np.ndarray:
    if scaler.method == "zscore":
        return (y - scaler.mean) / (scaler.std if scaler.std else 1.0)
    elif scaler.method == "asinh_then_zscore":
        s = scaler.asinh_scale if scaler.asinh_scale else 1.0
        y_tr = np.arcsinh(y / s)
        return (y_tr - scaler.mean) / (scaler.std if scaler.std else 1.0)
    else:
        raise ValueError(f"Unknown label method: {scaler.method}")


def inverse_transform_labels(y_trf: np.ndarray, scaler: LabelScaler) -> np.ndarray:
    if scaler.method == "zscore":
        return y_trf * (scaler.std if scaler.std else 1.0) + (scaler.mean if scaler.mean else 0.0)
    elif scaler.method == "asinh_then_zscore":
        s = scaler.asinh_scale if scaler.asinh_scale else 1.0
        y_asinh = y_trf * (scaler.std if scaler.std else 1.0) + (scaler.mean if scaler.mean else 0.0)
        return np.sinh(y_asinh) * s
    else:
        raise ValueError(f"Unknown label method: {scaler.method}")
    
    
# ---- Save and Load Scalars ---- #
def save_feature_scaler(scaler: FeatureScaler, features_dir: Path) -> None:
    out = features_dir / "scaler_features.json"
    with open(out, "w") as f:
        json.dump(scaler.to_json(), f, indent=2)


def save_label_scaler(scaler: LabelScaler, labels_dir: Path) -> None:
    """
    Save label scaler with BOTH canonical and legacy keys:
      - canonical: method, mean, std, asinh_scale
      - legacy   : method_y, train_mean, train_std
    This avoids ambiguity for any evaluator/plotter.
    """
    base = scaler.to_json()  # {'method','train_years','std_floor','mean','std','asinh_scale'}
    out_dict = dict(base)

    # write aliases too
    out_dict["method_y"] = base.get("method")
    if base.get("mean") is not None:
        out_dict["train_mean"] = float(base["mean"])
    if base.get("std") is not None:
        out_dict["train_std"] = float(base["std"])

    out = labels_dir / "scaler_labels.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)


def load_feature_scaler(features_dir: Path) -> FeatureScaler:
    p = features_dir / "scaler_features.json"
    with open(p, "r") as f:
        d = json.load(f)
    transforms = [FeatureTransformSpec(**t) for t in d["transforms"]]
    return FeatureScaler(
        train_years=d["train_years"],
        std_floor=d["std_floor"],
        zscore_after_log1p=d.get("zscore_after_log1p", False),
        drop_features=d["drop_features"],
        passthrough_features=d["passthrough_features"],
        log1p_features=d["log1p_features"],
        keep_feature_names=d["keep_feature_names"],
        transforms=transforms,
    )

def _nan_skew(x: np.ndarray) -> float:
    """Fisher-Pearson skewness ignoring NaNs: E[(x-μ)^3] / σ^3."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.nanmean(x))
    s = float(np.nanstd(x, ddof=0))
    if s == 0.0 or np.isnan(s):
        return float("nan")
    xc = x - m
    return float(np.nanmean(xc ** 3) / (s ** 3))


def load_label_scaler(labels_dir: Path) -> LabelScaler:
    p = labels_dir / "scaler_labels.json"
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    method = d.get("method") or d.get("method_y") or "zscore"
    # prefer canonical keys; fall back to aliases
    mean = d.get("mean", d.get("train_mean", None))
    std  = d.get("std",  d.get("train_std",  None))

    # Keep existing fields; default sensibly if missing
    train_years = d.get("train_years", [])
    std_floor   = float(d.get("std_floor", 1e-3))
    asinh_scale = d.get("asinh_scale", None)

    return LabelScaler(
        method=method,
        train_years=train_years,
        std_floor=std_floor,
        mean=(float(mean) if mean is not None else None),
        std=(float(std) if std is not None else None),
        asinh_scale=(float(asinh_scale) if asinh_scale is not None else None),
    )


# ---- Reporting ---- #
def report_normalization(
    features_dir: Path,
    labels_dir: Path,
    years_vec: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Prints a detailed Before/After table:
      - Train-slice mean/std BEFORE and AFTER transform
      - All-years min/max/skew BEFORE and AFTER
    Returns a dict with the computed stats.
    """
    fs = load_feature_scaler(features_dir)
    ls = load_label_scaler(labels_dir)

    # Build masks
    train_set = set(fs.train_years)
    t_mask = np.array([yr in train_set for yr in years_vec], dtype=bool)  # [T]

    # Transform features (AFTER)
    X_trf, kept = apply_feature_transform_inplace(X, feature_names, fs)
    # Map original names → original indices
    orig_idx = _index_map(feature_names)
    kept_to_orig = [orig_idx[nm] for nm in kept]

    # Flatten helpers
    def flat_train(arr):  # [T,N,...] -> [T*N]
        z = arr[t_mask]
        return z.reshape(-1, z.shape[-1]) if z.ndim == 3 else z.reshape(-1)

    def flat_all(arr):
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1)

    # BEFORE / AFTER, features
    X_train_flat_B = flat_train(X)                   # [(Ttr*N), F_orig]
    X_all_flat_B   = flat_all(X)                     # [(Tall*N), F_orig]
    X_train_flat_A = flat_train(X_trf)               # [(Ttr*N), F_kept]
    X_all_flat_A   = flat_all(X_trf)                 # [(Tall*N), F_kept]

    rows = []
    for j_after, spec in enumerate(fs.transforms):
        name = spec.name
        j_before = kept_to_orig[j_after]

        # BEFORE stats
        colB_tr = X_train_flat_B[:, j_before]
        colB_al = X_all_flat_B[:,   j_before]
        colB_tr = colB_tr[np.isfinite(colB_tr)]
        colB_al = colB_al[np.isfinite(colB_al)]

        mean_B = float(np.nanmean(colB_tr)) if colB_tr.size else float("nan")
        std_B  = float(np.nanstd(colB_tr, ddof=0)) if colB_tr.size else float("nan")
        min_B  = float(np.nanmin(colB_al)) if colB_al.size else float("nan")
        max_B  = float(np.nanmax(colB_al)) if colB_al.size else float("nan")
        skew_B = _nan_skew(colB_al)

        # AFTER stats
        colA_tr = X_train_flat_A[:, j_after]
        colA_al = X_all_flat_A[:,   j_after]
        colA_tr = colA_tr[np.isfinite(colA_tr)]
        colA_al = colA_al[np.isfinite(colA_al)]

        mean_A = float(np.nanmean(colA_tr)) if colA_tr.size else float("nan")
        std_A  = float(np.nanstd(colA_tr, ddof=0)) if colA_tr.size else float("nan")
        min_A  = float(np.nanmin(colA_al)) if colA_al.size else float("nan")
        max_A  = float(np.nanmax(colA_al)) if colA_al.size else float("nan")
        skew_A = _nan_skew(colA_al)

        rows.append({
            "name": name, "kind": spec.kind,
            "Mean_B": mean_B, "Std_B": std_B, "Min_B": min_B, "Max_B": max_B, "Skew_B": skew_B,
            "Mean_A": mean_A, "Std_A": std_A, "Min_A": min_A, "Max_A": max_A, "Skew_A": skew_A,
        })

    # Labels BEFORE / AFTER
    y_trf = transform_labels_inplace(y, ls)
    y_tr_B = y[t_mask].reshape(-1)
    y_al_B = y.reshape(-1)
    y_tr_A = y_trf[t_mask].reshape(-1)
    y_al_A = y_trf.reshape(-1)

    y_stats = {
        "method": ls.method,
        "Mean_B": float(np.nanmean(y_tr_B)) if np.isfinite(y_tr_B).any() else float("nan"),
        "Std_B":  float(np.nanstd(y_tr_B, ddof=0)) if np.isfinite(y_tr_B).any() else float("nan"),
        "Min_B":  float(np.nanmin(y_al_B)) if np.isfinite(y_al_B).any() else float("nan"),
        "Max_B":  float(np.nanmax(y_al_B)) if np.isfinite(y_al_B).any() else float("nan"),
        "Skew_B": _nan_skew(y_al_B),
        "Mean_A": float(np.nanmean(y_tr_A)) if np.isfinite(y_tr_A).any() else float("nan"),
        "Std_A":  float(np.nanstd(y_tr_A, ddof=0)) if np.isfinite(y_tr_A).any() else float("nan"),
        "Min_A":  float(np.nanmin(y_al_A)) if np.isfinite(y_al_A).any() else float("nan"),
        "Max_A":  float(np.nanmax(y_al_A)) if np.isfinite(y_al_A).any() else float("nan"),
        "Skew_A": _nan_skew(y_al_A),
        "asinh_scale": ls.asinh_scale if getattr(ls, "asinh_scale", None) else None,
    }

    out = {"features": rows, "labels": y_stats}

    if verbose:
        # Pretty print table similar to your old output
        header = (
            f"{'Feature':<20s} {'Kind':<18s}"
            f"{'Mean_B':>10s} {'Std_B':>8s} {'Min_B':>10s} {'Max_B':>10s} {'Skew_B':>7s}   "
            f"{'Mean_A':>10s} {'Std_A':>8s} {'Min_A':>10s} {'Max_A':>10s} {'Skew_A':>7s}"
        )
        print("\nFeature stats (Before/After; train means/stds, all-years min/max/skew):")
        print(header)
        for r in rows:
            print(
                f"{r['name']:<20s} {r['kind']:<18s}"
                f"{r['Mean_B']:>10.3g} {r['Std_B']:>8.3g} {r['Min_B']:>10.3g} {r['Max_B']:>10.3g} {r['Skew_B']:>7.2f}   "
                f"{r['Mean_A']:>10.3g} {r['Std_A']:>8.3g} {r['Min_A']:>10.3g} {r['Max_A']:>10.3g} {r['Skew_A']:>7.2f}"
            )
        print("\nLabel stats (Before/After; train means/stds, all-years min/max/skew):")
        print(
            f"  method={y_stats['method']}, "
            f"Mean_B={y_stats['Mean_B']:.3g}, Std_B={y_stats['Std_B']:.3g}, "
            f"Min_B={y_stats['Min_B']:.3g}, Max_B={y_stats['Max_B']:.3g}, Skew_B={y_stats['Skew_B']:.2f}; "
            f"Mean_A={y_stats['Mean_A']:.3g}, Std_A={y_stats['Std_A']:.3g}, "
            f"Min_A={y_stats['Min_A']:.3g}, Max_A={y_stats['Max_A']:.3g}, Skew_A={y_stats['Skew_A']:.2f}"
            + (f", asinh_scale={y_stats['asinh_scale']:.3g}" if y_stats.get("asinh_scale") else "")
        )

    return out

# ---- Driver ---- #
def run_normalization(cfg_path: Path) -> None:
    ROOT = cfg_path.resolve().parents[3]  # adjust if your layout differs
    cfg = _load_yaml(cfg_path)
    city = cfg["dataset"]["city"]
    paths = _resolve_paths_from_yaml(ROOT, city)

    norm = cfg["normalize"]
    train_end_year = int(norm["train_end_year"])
    std_floor = float(norm.get("std_floor", 1e-3))
    zscore_after_log1p = bool(norm.get("zscore_after_log1p", False))

    log1p_features = norm.get("log1p_features", [])
    passthrough_features = norm.get("passthrough_features", [])
    drop_features = norm.get("drop_features", [])

    method_y = norm.get("method_y", "zscore")
    asinh_scale = norm.get("asinh_scale", "auto")

    # Load arrays
    X = np.load(paths["features_dir"] / "X.npy", mmap_mode="r")
    mask = np.load(paths["features_dir"] / "mask.npy", mmap_mode="r")
    y = np.load(paths["labels_dir"] / "y.npy", mmap_mode="r")
    y_mask = np.load(paths["labels_dir"] / "y_mask.npy", mmap_mode="r")
    feature_names = _load_feature_names(paths["features_dir"])
    years_vec, _ = _years_vector(paths["features_dir"])

    # Fit scalers
    fs = fit_feature_scaler(
        X=X, mask=mask, years_vec=years_vec, feature_names=feature_names,
        train_end_year=train_end_year, std_floor=std_floor,
        log1p_features=log1p_features,
        passthrough_features=passthrough_features,
        drop_features=drop_features,
        zscore_after_log1p=zscore_after_log1p,
    )
    ls = fit_label_scaler(
        y=y, y_mask=y_mask, years_vec=years_vec,
        train_end_year=train_end_year, std_floor=std_floor,
        method=method_y, asinh_scale=asinh_scale,
    )

    # Save scalers
    save_feature_scaler(fs, paths["features_dir"])
    save_label_scaler(ls, paths["labels_dir"])

    # Optional report
    if bool(norm.get("run_report_on_fit", True)):
        report_normalization(
            features_dir=paths["features_dir"],
            labels_dir=paths["labels_dir"],
            years_vec=years_vec,
            X=X,
            y=y,
            feature_names=feature_names,
            verbose=True,
        )


# ---------------------------- CLI ----------------------------

# if __name__ == "__main__":
#     # Example CLI run:
#     #   PYTHONPATH=. python RNN/prep/normalize_city.py --cfg RNN/config/default.yaml
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config (default.yaml)")
#     args = ap.parse_args()
#     run_normalization(Path(args.cfg))

