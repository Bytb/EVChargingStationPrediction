# EV_Research/RNN/training/callbacks.py
from __future__ import annotations
import math
from pathlib import Path
import torch

class EarlyStopper:
    def __init__(self, patience: int = 20, mode: str = "min", min_delta: float = 0.0):
        assert mode in ("min","max")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = math.inf if mode=="min" else -math.inf
        self.counter = 0
        self.stopped = False

    def step(self, value: float) -> bool:
        improved = (value < self.best - self.min_delta) if self.mode=="min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
        return False

class CheckpointManager:
    def __init__(self, out_dir: Path):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.best_path = self.out / "best.ckpt"
        self.last_path = self.out / "last.ckpt"
    def save(self, model, optimizer, epoch: int, metric_value: float, is_best: bool):
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metric": metric_value}
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)
