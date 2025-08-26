# RNN/prep/test_normalize.py
from pathlib import Path
import yaml
import numpy as np

# Import the Step 2.5 helpers (we call functions; we do NOT run normalize_city.py as a script)
from Normalize_City import (
    run_normalization,
    load_feature_scaler,
    load_label_scaler,
    apply_feature_transform_inplace,
    transform_labels_inplace,
    inverse_transform_labels,
    _load_feature_names,
    _years_vector,
)

def find_root(start: Path, marker=("RNN", "config", "default.yaml"), max_hops=6) -> Path:
    """Walk upward until we find the project root that contains RNN/config/default.yaml."""
    p = start
    for _ in range(max_hops):
        if (p / marker[0] / marker[1] / marker[2]).exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate project root. Adjust `max_hops` or your path structure.")

def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # 1) Resolve ROOT and CFG
    HERE = Path(__file__).resolve()
    ROOT = find_root(HERE)
    CFG_PATH = ROOT / "RNN" / "config" / "default.yaml"
    PROJECT   = ROOT.parent                      # <-- this is EV_Research_PythonCode
    print(f"[test_normalize] Project root: {ROOT}")
    print(f"[test_normalize] Config:       {CFG_PATH}")

    cfg = load_cfg(CFG_PATH)
    city = cfg["dataset"]["city"]
    city_root = PROJECT / "data" / city
    features_dir = city_root / "features_RNN"
    labels_dir   = city_root / "labels_RNN"

    # 2) Run Step 2.5 normalization (this writes the scalers + prints a report)
    print("\n[test_normalize] Running Step 2.5 normalization...")
    run_normalization(CFG_PATH)

    # 3) Confirm scaler files exist
    fe_scaler_path = features_dir / "scaler_features.json"
    lb_scaler_path = labels_dir   / "scaler_labels.json"
    ok_files = fe_scaler_path.exists() and lb_scaler_path.exists()
    print(f"[test_normalize] Scaler files exist: {ok_files}")
    if not ok_files:
        raise FileNotFoundError("Expected scaler_features.json or scaler_labels.json not found.")

    # 4) Load raw arrays & metadata
    print("[test_normalize] Loading arrays...")
    X = np.load(features_dir / "X.npy", mmap_mode="r")     # [T,N,F]
    y = np.load(labels_dir   / "y.npy", mmap_mode="r")     # [T,N]
    feature_names = _load_feature_names(features_dir)
    years_vec, _ = _years_vector(features_dir)

    # 5) Load scalers and re-apply transforms in memory
    fs = load_feature_scaler(features_dir)
    ls = load_label_scaler(labels_dir)

    X_trf, kept_names = apply_feature_transform_inplace(X, feature_names, fs)
    print(f"[test_normalize] X shape {X.shape}  ->  X_trf shape {X_trf.shape}")
    print(f"[test_normalize] Kept features (first 10): {kept_names[:10]}")

    y_trf = transform_labels_inplace(y, ls)
    y_back = inverse_transform_labels(y_trf, ls)
    # NaN-safe max abs error for round-trip
    mask = np.isfinite(y_back) & np.isfinite(y)
    max_err = float(np.max(np.abs((y_back - y)[mask]))) if mask.any() else 0.0
    print(f"[test_normalize] Label inverse-transform max |error|: {max_err:.6g}")

    # 6) Train-slice stats check (similar to the built-in report), but brief
    train_years = set(fs.train_years)
    t_mask = np.array([yr in train_years for yr in years_vec], dtype=bool)
    X_trf_train = X_trf[t_mask].reshape(-1, X_trf.shape[-1])  # [(Ttrain*N), F_kept]
    y_trf_train = y_trf[t_mask].reshape(-1)

    # For z-scored features, mean≈0 std≈1 (we don't know which are z-scored vs log1p/passthrough, so look at fs.transforms)
    mean_std_checks = []
    for j, spec in enumerate(fs.transforms):
        if spec.kind in ("zscore", "log1p_then_zscore"):
            col = X_trf_train[:, j]
            col = col[np.isfinite(col)]
            if col.size:
                m = float(np.nanmean(col))
                s = float(np.nanstd(col, ddof=0))
                mean_std_checks.append((spec.name, m, s))

    # print a few
    print("\n[test_normalize] Train-slice z-scored features (sample):")
    for name, m, s in mean_std_checks[:10]:
        print(f"  - {name:30s} mean≈{m:+.4f}  std≈{s:.4f}")

    # Label z-score/asinh_then_zscore mean/std check
    y_m = float(np.nanmean(y_trf_train)) if y_trf_train.size else 0.0
    y_s = float(np.nanstd(y_trf_train, ddof=0)) if y_trf_train.size else 0.0
    print(f"\n[test_normalize] Label train-slice (transformed) mean≈{y_m:+.4f}  std≈{y_s:.4f}  method={ls.method}")

    # 7) Final PASS/FAIL summary
    pass_roundtrip = max_err < 1e-5
    # Loose tolerances; we expect near 0/1 but allow small drift
    pass_label_stats = (abs(y_m) < 0.05) and (0.90 <= y_s <= 1.10)
    # if there are any z-scored features, check first few
    pass_feat_stats = True
    for _, m, s in mean_std_checks[:5]:
        if not (abs(m) < 0.05 and 0.85 <= s <= 1.15):
            pass_feat_stats = False
            break

    all_ok = ok_files and pass_roundtrip and pass_label_stats and pass_feat_stats
    print(f"\n[test_normalize] PASSED: {all_ok}")
    if not all_ok:
        print("  Details:")
        print(f"    - scaler files present: {ok_files}")
        print(f"    - label inverse round-trip ok (<1e-5): {pass_roundtrip}")
        print(f"    - label mean/std within tolerance:    {pass_label_stats}")
        print(f"    - feature mean/std within tolerance:  {pass_feat_stats}")
