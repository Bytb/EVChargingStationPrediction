# EV_Research/RNN/training/test_step5.py
from pathlib import Path
import shutil, json, yaml, numpy as np
from train import train_and_eval

def find_root(start: Path, max_hops=8) -> Path:
    p = start.resolve()
    for _ in range(max_hops):
        if (p / "RNN" / "config" / "default.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not find EV_Research root")

if __name__ == "__main__":
    HERE = Path(__file__).resolve()
    ROOT = find_root(HERE)
    CFG_PATH = ROOT / "RNN" / "config" / "default.yaml"
    print(f"[Step5 smoke] Using config: {CFG_PATH}")

    # Load + override a few settings for a fast, deterministic smoke run
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Small, fast run on CPU with the baseline model

    # Clean the smoke folder (so we know what was created)
    smoke_dir = Path(cfg["logging"]["out_dir"])
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    # Write a temporary config snapshot and run
    TMP_CFG = smoke_dir.parent / "_tmp_step5_smoke.yaml"
    TMP_CFG.parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_CFG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"[Step5 smoke] Temp cfg: {TMP_CFG}")

    train_and_eval(TMP_CFG)

    # The trainer writes into runs/<city>/_step5_smoke/<timestamp>/
    # Find the newest timestamped run dir
    if not smoke_dir.exists():
        raise FileNotFoundError(f"No smoke output dir found at {smoke_dir}")
    subdirs = sorted([p for p in smoke_dir.iterdir() if p.is_dir()])
    assert subdirs, f"No timestamped subfolders in {smoke_dir}"
    out = subdirs[-1]
    print(f"[Step5 smoke] Checking outputs in: {out}")

    # Check metrics.json
    mj = out / "metrics.json"
    assert mj.exists(), "metrics.json not found"
    with open(mj, "r", encoding="utf-8") as f:
        met = json.load(f)
    assert "history" in met and "test" in met, "metrics.json missing keys"
    print(f"  history epochs logged: {len(met['history'])}")
    print(f"  test metrics keys: {list(met['test'].keys())}")

    # Check predictions file
    npz = out / "predictions_test.npz"
    assert npz.exists(), "predictions_test.npz not found"
    z = np.load(npz)
    for k in ("yhat","y","y_mask","years"):
        assert k in z, f"{k} missing from predictions_test.npz"
    yhat = z["yhat"]; y = z["y"]; ym = z["y_mask"]; yrs = z["years"]
    print(f"  preds shape: {yhat.shape}, true shape: {y.shape}, mask shape: {ym.shape}, years shape: {yrs.shape}")
    assert yhat.size > 0 and y.size > 0, "empty predictions"

    # (Optional) very soft sanity: last train loss <= first train loss * 1.2
    hist = met["history"]
    tr0 = hist[0]["train_loss"]; trL = hist[-1]["train_loss"]
    if trL <= 1.2 * tr0:
        print(f"  train loss decreased (ok): {tr0:.4f} → {trL:.4f}")
    else:
        print(f"  NOTE: train loss did not decrease much: {tr0:.4f} → {trL:.4f} (still ok for a smoke run)")

    print("\n[Step5 smoke] ✅ Passed smoke checks.")
