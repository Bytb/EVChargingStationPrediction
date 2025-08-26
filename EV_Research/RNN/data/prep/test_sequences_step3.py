# EV_Research/RNN/data/test_sequences.py
from pathlib import Path
import yaml

from Loaders import check_sequences, EVSequenceDataset, make_dataloader
from Splits import build_split_indices, describe_splits

def find_root(start: Path, max_hops=6) -> Path:
    p = start.resolve()
    for _ in range(max_hops):
        if (p / "RNN" / "config" / "default.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Could not locate EV_Research root.")

if __name__ == "__main__":
    HERE = Path(__file__).resolve()
    EV_RESEARCH_ROOT = find_root(HERE)               # .../EV_Research
    CFG_PATH = EV_RESEARCH_ROOT / "RNN" / "config" / "default.yaml"
    print(f"[test_sequences] Config: {CFG_PATH}")

    # ---- Step 3: windowing / transforms / adjacency check ----
    check_sequences(CFG_PATH, examples=10)

    # You also need the dataset + YAML loaded for Step 4:
    ds = EVSequenceDataset(CFG_PATH)
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- Step 4: forward-chained splits by target year ----
    train_end_year = int(cfg["normalize"]["train_end_year"])
    idx = build_split_indices(ds.target_years, train_end_year)

    print("\n[Step 4] Split summary:")
    print(describe_splits(ds.target_years, idx))

    # Optional: peek one example window from each split
    for split in ("train", "val", "test"):
        if len(idx[split]) == 0:
            print(f"  No samples in {split}.")
            continue
        i = int(idx[split][0])               # sample index within ds
        t = int(ds.targets_idx[i])           # target time index in 0..T-1
        L = int(cfg.get("dataset", {}).get("window_L", 4))
        yrs_in = ds.years_vec[t - L:t].tolist()
        yr_out = int(ds.years_vec[t])
        print(f"  {split} example: input {yrs_in} -> target {yr_out}")

    # Optional: verify DataLoaders use those indices correctly
    for split in ("train", "val", "test"):
        try:
            loader, ds2, _ = make_dataloader(CFG_PATH, split=split, batch_size=2, shuffle=False)
            n_batches = sum(1 for _ in loader)
            print(f"  {split} DataLoader: {len(loader.dataset)} samples, {n_batches} batches")
        except Exception as e:
            print(f"  {split} DataLoader error: {e}")
