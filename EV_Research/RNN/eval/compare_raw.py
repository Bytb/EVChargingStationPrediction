# RNN/eval/compare_raw.py
from __future__ import annotations
from pathlib import Path
import argparse, json, yaml, numpy as np
from math import sqrt

def _latest_run(runs_root: Path) -> Path:
    # pick newest leaf dir under runs_root/*/*/<timestamp>
    candidates = [p for p in runs_root.rglob("*") if (p / "predictions_test.npz").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run folders with predictions_test.npz under {runs_root}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.as_posix()))

def _load_cfg_for_run(run_dir: Path) -> dict:
    snap = run_dir / "config.snapshot.yaml"
    if snap.exists():
        return yaml.safe_load(snap.read_text(encoding="utf-8"))
    # fallback: try default.yaml via root sniff
    p = run_dir
    for _ in range(6):
        guess = p / "RNN" / "config" / "default.yaml"
        if guess.exists():
            return yaml.safe_load(guess.read_text(encoding="utf-8"))
        p = p.parent
    raise FileNotFoundError("Could not locate config for run.")

def _load_label_scaler(city_root: Path) -> dict:
    f = city_root / "labels_RNN" / "scaler_labels.json"
    if not f.exists():
        raise FileNotFoundError(f"Missing label scaler: {f}")
    return json.loads(f.read_text(encoding="utf-8"))

def invert_labels(z: np.ndarray, scaler: dict) -> np.ndarray:
    # be robust to slightly different key names
    method = (scaler.get("method") or scaler.get("method_y") or "zscore").lower()
    mean = float(scaler.get("train_mean") or scaler.get("mean"))
    std  = float(scaler.get("train_std")  or scaler.get("std")  or 1.0)
    if method == "zscore":
        return z * std + mean
    elif method in ("asinh_then_zscore", "asinh"):
        scale = float(scaler.get("asinh_scale") or scaler.get("scale") or 1.0)
        raw_std = z * std + mean
        return np.sinh(raw_std) * scale
    else:
        raise ValueError(f"Unknown label method: {method}")

def masked_metrics(yhat: np.ndarray, y: np.ndarray, m: np.ndarray) -> dict:
    m = m.astype(bool)
    yhat_m = yhat[m]; y_m = y[m]
    err = yhat_m - y_m
    mae = float(np.mean(np.abs(err)))
    rmse = float(sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    return dict(mae=mae, rmse=rmse, bias=bias, n=int(m.sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="", help="Path to run dir containing predictions_test.npz")
    ap.add_argument("--runs_root", type=str, default="runs", help="Root to search if --run not given")
    ap.add_argument("--plot", action="store_true", help="Show scatter & hist in raw scale")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else _latest_run(Path(args.runs_root))
    pred_f  = run_dir / "predictions_test.npz"
    if not pred_f.exists():
        raise FileNotFoundError(f"Not found: {pred_f}")

    cfg = _load_cfg_for_run(run_dir)
    city = cfg["dataset"]["city"]
    city_root = Path("data") / city  # matches your layout data/<city>/*
    scaler = _load_label_scaler(city_root)

    z = np.load(pred_f)
    yhat_std = z["yhat"]       # [B,N]
    y_std    = z["y"]          # [B,N]
    y_mask   = z["y_mask"]     # [B,N]
    years    = z["years"]      # [B]

    # Invert to RAW scale
    yhat_raw = invert_labels(yhat_std, scaler)
    y_raw    = invert_labels(y_std,  scaler)

    # quick drift check (raw scale)
    m = y_mask.astype(bool)
    train_mu = float(scaler.get("mean") or scaler.get("train_mean"))
    test_mu  = float((y_raw[m]).mean())
    print(f"  Train mean (raw) ≈ {train_mu:.3f}   Test mean (raw) ≈ {test_mu:.3f}")


    # Metrics per test batch (usually B==1), and overall
    all_mets = []
    for i in range(yhat_raw.shape[0]):
        mets = masked_metrics(yhat_raw[i], y_raw[i], y_mask[i])
        all_mets.append((int(years[i]), mets))

    # Print summary
    print(f"\nRun: {run_dir}")
    print(f"City: {city}")
    print(f"Label scaler: method={scaler.get('method') or scaler.get('method_y')} "
          f"mean={scaler.get('train_mean')} std={scaler.get('train_std')} "
          f"asinh_scale={scaler.get('asinh_scale')}")
    for yr, m in all_mets:
        print(f"  Test year {yr}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  Bias={m['bias']:.3f}  (n={m['n']})")

        # Extra summary
    print("\nExtra summary (raw):")
    for i in range(yhat_raw.shape[0]):
        m = y_mask[i].astype(bool)
        print(f"  batch {i}: mean_true={float(y_raw[i][m].mean()):.3f}  "
              f"mean_pred={float(yhat_raw[i][m].mean()):.3f}  "
              f"mean_err={float((yhat_raw[i][m]-y_raw[i][m]).mean()):.3f}")

    # Optional: plots similar to your GCN script (raw scale)
    if args.plot:
        import matplotlib.pyplot as plt
        y_true = y_raw.reshape(-1)
        y_pred = yhat_raw.reshape(-1)
        m = y_mask.reshape(-1).astype(bool)
        y_true = y_true[m]; y_pred = y_pred[m]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Scatter
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=16)
        mn, mx = float(np.min(y_true)), float(np.max(y_true))
        axes[0].plot([mn, mx], [mn, mx], lw=2)
        axes[0].set_title("True vs Predicted (raw)")
        axes[0].set_xlabel("True"); axes[0].set_ylabel("Pred")
        axes[0].grid(True)
        # Histogram overlay
        axes[1].hist(y_true, bins=30, alpha=0.6, label="True")
        axes[1].hist(y_pred, bins=30, alpha=0.6, label="Pred")
        axes[1].set_title("Label vs Prediction Distribution (raw)")
        axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
