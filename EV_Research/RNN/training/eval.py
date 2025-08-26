# EV_Research/RNN/training/eval.py
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

def _mask_numpy(arr: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = m.astype(bool)
    return arr[m]

def mae(yhat, y, m): return np.mean(np.abs(_mask_numpy(yhat - y, m)))
def mse(yhat, y, m): return np.mean((_mask_numpy(yhat - y, m)) ** 2)
def rmse(yhat, y, m): return np.sqrt(mse(yhat, y, m))
def bias(yhat, y, m): return np.mean(_mask_numpy(yhat - y, m))
def under_bias(yhat, y, m): return np.mean(_mask_numpy(np.clip(y - yhat, 0, None), m))
def over_bias(yhat, y, m):  return np.mean(_mask_numpy(np.clip(yhat - y, 0, None), m))

def compute_metrics(yhat_t: torch.Tensor, y_t: torch.Tensor, m_t: torch.Tensor,
                    surge_threshold: float | None = None) -> Dict[str, float]:
    # yhat_t, y_t, m_t: [B,N] tensors
    yhat = yhat_t.detach().cpu().numpy()
    y = y_t.detach().cpu().numpy()
    m = m_t.detach().cpu().numpy().astype(bool)

    out = {
        "mae":  mae(yhat, y, m),
        "rmse": rmse(yhat, y, m),
        "bias": bias(yhat, y, m),
        "under_bias": under_bias(yhat, y, m),
        "over_bias":  over_bias(yhat, y, m),
    }

    if surge_threshold is not None:
        surge_mask = (y >= surge_threshold)
        if surge_mask.any():
            ms = m & surge_mask
            out.update({
                "mae_surge":  mae(yhat, y, ms),
                "rmse_surge": rmse(yhat, y, ms),
                "bias_surge": bias(yhat, y, ms),
            })
        else:
            out.update({"mae_surge": np.nan, "rmse_surge": np.nan, "bias_surge": np.nan})
    return out
