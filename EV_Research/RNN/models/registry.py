# EV_Research/RNN/models/registry.py
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .base import BaseSpatioTemporalModel

class LastYearMLP(BaseSpatioTemporalModel):
    """
    Simple baseline: take last year's features per node and predict y via MLP.
    """
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, X_seq, A, mask_seq):
        # X_seq: [B,L,N,F] -> use last step X_seq[:, -1, :, F]
        x_last = X_seq[:, -1]           # [B,N,F]
        B,N,F = x_last.shape
        out = self.net(x_last.reshape(B*N, F)).reshape(B, N)
        return out

# New models
from .gwnet import GraphWaveNet as _GWNET
from .influencer_rank import InfluencerRank as _INF

def make_model(name: str, n_features: int, n_nodes: int, **kwargs) -> BaseSpatioTemporalModel:
    """
    name: 'baseline' | 'gwnet' | 'influencer_rank'
    n_features: feature dim per node
    n_nodes:    number of nodes (for adaptive adjacency, etc.)
    kwargs:     model-specific hyperparams from YAML under `model:`
    """
    name = (name or "baseline").lower()
    if name in ("baseline", "lastyear_mlp"):
        return LastYearMLP(n_features=n_features, **kwargs)
    if name in ("gwnet", "graphwavenet"):
        return _GWNET(
            n_nodes=n_nodes,
            n_features=n_features,
            **kwargs
        )
    if name in ("influencer_rank","influencerrank","infrank"):
        return _INF(
            n_nodes=n_nodes,
            n_features=n_features,
            **kwargs
        )
    raise ValueError(f"Unknown model: {name}")