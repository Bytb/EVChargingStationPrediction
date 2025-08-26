# EV_Research/RNN/training/loss.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

def _masked_reduce(loss: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    # loss, y_mask: [B,N]
    w = y_mask.float()
    s = (loss * w).sum()
    n = w.sum().clamp_min(1.0)
    return s / n

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)
    def forward(self, pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        e = pred - target
        abs_e = e.abs()
        quad = 0.5 * (e ** 2)
        lin  = self.delta * (abs_e - 0.5 * self.delta)
        loss = torch.where(abs_e <= self.delta, quad, lin)
        return _masked_reduce(loss, y_mask)

class MaskedQuantileLoss(nn.Module):
    def __init__(self, q: float = 0.8):
        super().__init__()
        assert 0.0 < q < 1.0
        self.q = float(q)
    def forward(self, pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        e = target - pred  # pinball uses (y - yhat)
        loss = torch.maximum(self.q * e, (self.q - 1.0) * e)  # [B,N]
        return _masked_reduce(loss, y_mask)

class MaskedUnderpredictionHuber(nn.Module):
    """Huber where under-prediction (pred < target) gets extra weight."""
    def __init__(self, delta: float = 1.0, under_weight: float = 2.0):
        super().__init__()
        self.delta = float(delta)
        self.under_w = float(under_weight)
    def forward(self, pred: torch.Tensor, target: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        e = pred - target
        abs_e = e.abs()
        huber = torch.where(abs_e <= self.delta, 0.5 * e * e, self.delta * (abs_e - 0.5 * self.delta))
        w = torch.ones_like(huber)
        w = torch.where(pred < target, self.under_w * w, w)  # under-pred heavier
        return _masked_reduce(huber * w, y_mask)

def get_loss(cfg_loss: Dict) -> nn.Module:
    kind = (cfg_loss.get("kind", "huber") or "huber").lower()
    if kind == "huber":
        return MaskedHuberLoss(delta=float(cfg_loss.get("huber_delta", 1.0)))
    if kind == "quantile":
        return MaskedQuantileLoss(q=float(cfg_loss.get("quantile_q", 0.8)))
    if kind == "underpred_huber":
        return MaskedUnderpredictionHuber(
            delta=float(cfg_loss.get("huber_delta", 1.0)),
            under_weight=float(cfg_loss.get("underpred_weight", 2.0)),
        )
    raise ValueError(f"Unknown loss kind: {kind}")
