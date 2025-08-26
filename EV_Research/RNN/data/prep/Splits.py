# EV_Research/RNN/data/splits.py
from __future__ import annotations
import numpy as np
from typing import Dict

def build_split_masks(
    years_targets: np.ndarray,  # [B] int years for each sample target
    train_end_year: int,
) -> Dict[str, np.ndarray]:
    """
    Build boolean masks for train/val/test based on the target year of each sample.
    - Train: target_year <= train_end_year
    - Val:   target_year == train_end_year + 1 (if present)
    - Test:  target_year > train_end_year + 1
    """
    years_targets = np.asarray(years_targets)
    val_year = train_end_year + 1

    train_mask = years_targets <= train_end_year
    val_mask   = years_targets == val_year
    test_mask  = years_targets > val_year

    return dict(train=train_mask, val=val_mask, test=test_mask)

def build_split_indices(years_targets: np.ndarray, train_end_year: int) -> Dict[str, np.ndarray]:
    m = build_split_masks(years_targets, train_end_year)
    return {k: np.flatnonzero(v) for k, v in m.items()}

def describe_splits(years_targets: np.ndarray, idx: Dict[str, np.ndarray]) -> str:
    lines = []
    lines.append(f"All target years: {years_targets.tolist()}")
    for k in ("train", "val", "test"):
        yrs = years_targets[idx[k]].tolist()
        lines.append(f"{k:>5}: years={yrs}  count={len(yrs)}")
    return "\n".join(lines)
