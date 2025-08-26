# EV_Research/RNN/models/influencer_rank.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSpatioTemporalModel

def _gcn_norm(A: torch.Tensor, add_self: bool = True) -> torch.Tensor:
    if add_self:
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    deg = A.sum(-1).clamp(min=1e-6)
    d = deg.pow(-0.5)
    return d[:, None] * A * d[None, :]

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X: torch.Tensor, A_hat: torch.Tensor):
        A_hat = A_hat.to(dtype=X.dtype, device=X.device)
        X = self.dropout(X)
        Z = torch.einsum("bnf,nm->bmf", X, A_hat)
        Z = self.lin(Z)
        return F.relu(Z)

class TemporalAttention(nn.Module):
    """Additive attention over time."""
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
    def forward(self, H: torch.Tensor, mask: torch.Tensor | None = None):
        # H: [B*N, L, H], mask: [B,N,L] or None
        scores = torch.tanh(self.proj(H))           # [B*N, L, H]
        scores = self.v(scores).squeeze(-1)         # [B*N, L]
        if mask is not None:
            m = mask.reshape(-1, mask.size(-1))     # [B*N, L]
            scores = scores.masked_fill(~m, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)       # [B*N, L]
        ctx = torch.einsum("bl,blh->bh", alpha, H)  # [B*N, H]
        return ctx, alpha

class InfluencerRank(BaseSpatioTemporalModel):
    """
    EV adaptation of InfluencerRank:
      - Per-time GCN encodes node features with road graph
      - GRU over time per node
      - Attention over time
      - MLP head per node → Δ prediction
    """
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        gcn_hidden: int = 64,
        gcn_layers: int = 2,
        gru_hidden: int = 64,
        attn_hidden: int | None = None,   # keep same as gru_hidden if None
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.dropout = nn.Dropout(dropout)

        # GCN stack (shared across time)
        ch = n_features
        gcns = []
        for i in range(gcn_layers):
            out = gcn_hidden if i < gcn_layers - 1 else gcn_hidden
            gcns.append(GCNLayer(ch, out, dropout=dropout))
            ch = out
        self.gcns = nn.ModuleList(gcns)

        # Temporal GRU across L time steps (per node)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)

        # after: self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)

        if (attn_hidden is not None) and (attn_hidden != gru_hidden):
            raise ValueError(f"attn_hidden must equal gru_hidden (got {attn_hidden} vs {gru_hidden}).")

        self.attn = TemporalAttention(hidden=gru_hidden)

        # Head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1),
        )

        # cache for normalized adjacency
        self._A_hat = None

    def forward(self, X_seq: torch.Tensor, A: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        """
        X_seq:   [B, L, N, F]
        A:       [N, N] (un-normalized or GCN-normalized; we normalize here)
        mask_seq:[B, L, N] (True where valid)
        """
        B, L, N, F = X_seq.shape
        assert N == self.n_nodes and F == self.n_features

        # before building A_hat
        if A.dim() == 3:
            A = A[0]
        A = A.to(device=X_seq.device, dtype=X_seq.dtype)

        # cache (also check dtype)
        # A_hat = self._A_hat
        # if (A_hat is None) or (A_hat.size(0) != N) or (A_hat.device != A.device) or (A_hat.dtype != A.dtype):
        #     A_hat = _gcn_norm(A, add_self=True)
        #     self._A_hat = A_hat
        # # ensure dtype matches X even if cached
        A_hat = A.to(dtype=X_seq.dtype)

        # Per-time GCN encoding
        # We'll loop over time for clarity (L is tiny: 2–5)
        H_t_list = []
        for t in range(L):
            Xt = X_seq[:, t, :, :]                 # [B,N,F]
            Ht = Xt
            for g in self.gcns:
                Ht = g(Ht, A_hat)                  # [B,N,gcn_hidden]
            H_t_list.append(Ht)
        # Stack time: [B,L,N,H]
        H = torch.stack(H_t_list, dim=1)

        # GRU per node → reshape [B,N,L,H] → [B*N, L, H]
        H_bn = H.permute(0, 2, 1, 3).contiguous().view(B * N, L, -1)
        # Build mask for attention (GRU handles packed sequences poorly for uniform L; we keep it simple.)
        m_seq = mask_seq.bool() if mask_seq is not None else None
        m_bn = m_seq.permute(0, 2, 1).contiguous().view(B * N, L) if m_seq is not None else None

        H_out, _ = self.gru(H_bn)                  # [B*N, L, gru_hidden]

        # Attention over time (mask locates valid timesteps; typically all True here)
        ctx, alpha = self.attn(H_out, mask=m_bn)   # [B*N, gru_hidden]

        # Head → [B*N, 1] → [B,N]
        yhat = self.head(ctx).view(B, N)
        return yhat
