# EV_Research/RNN/models/base.py
from __future__ import annotations
import torch
import torch.nn as nn

class BaseSpatioTemporalModel(nn.Module):
    """
    Expected forward signature:
      forward(X_seq, A, mask_seq) -> yhat [B,N]  (in label's transformed space)
      X_seq:  [B,L,N,F], A: [N,N], mask_seq: [B,L,N]
    """
    pass
