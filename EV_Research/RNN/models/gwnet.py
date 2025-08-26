# EV_Research/RNN/models/gwnet.py
from __future__ import annotations
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSpatioTemporalModel

# ---- small utilities ----
class Chomp1d(nn.Module):
    """Trim causal padding at the end along time."""
    def __init__(self, chomp: int): super().__init__(); self.chomp = chomp
    def forward(self, x):  # x: [B, C, T]
        return x[:, :, :-self.chomp] if self.chomp > 0 and x.size(-1) > self.chomp else x

class ConvTemporalGLU(nn.Module):
    """
    Causal, dilated 1D temporal conv (GLU style): split -> tanh/sigmoid.
    Expects [B, C_in, T]; keeps length with padding=(k-1)*d.
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, 2 * c_out, kernel_size,
                              dilation=dilation, padding=pad)
        self.chomp = Chomp1d(pad)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, C_in, T]
        x = self.conv(x)
        x = self.chomp(x)
        c_out = x.size(1) // 2
        x_tanh = torch.tanh(x[:, :c_out, :])
        x_sig  = torch.sigmoid(x[:, c_out:, :])
        return self.dropout(x_tanh * x_sig)  # [B, C_out, T]

class NConv(nn.Module):
    """Neighborhood conv via support matrices: einsum over nodes."""
    def forward(self, x, A):  # x:[B,C,N,T], A:[N,N]
        return torch.einsum("bcnt,nm->bcmt", x, A.contiguous())

class GcnMix(nn.Module):
    """
    Mix multiple supports (static A, A^T, and optional adaptive A_adp, A_adp^T)
    with 1x1 conv across concatenated results.
    """
    def __init__(self, c_in: int, c_out: int, n_supports: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in * n_supports, c_out, kernel_size=(1, 1))

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        # x_i: [B,C,N,T] with same shapes
        x = torch.cat(x_list, dim=1)
        return self.mlp(x)

class GraphWaveNet(BaseSpatioTemporalModel):
    """
    Compact Graph WaveNet:
      - Input: X_seq [B, L, N, F], A [N,N] (row-stochastic recommended), mask_seq ignored here
      - Output: yhat [B, N] (predict next-step per node; we take last-time skip path)
    """
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        out_dim: int = 1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        blocks: int = 2,
        layers_per_block: int = 3,
        dropout: float = 0.3,
        use_adaptive_adj: bool = True,
        adaptive_embed_dim: int = 10,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.use_adp = use_adaptive_adj

        # Start projection (1x1 over feature dim)
        self.start = nn.Conv2d(in_channels=n_features, out_channels=residual_channels, kernel_size=(1, 1))

        # Diffusion conv helpers
        self.nconv = NConv()

        self.blocks = blocks
        self.layers_per_block = layers_per_block
        self.kernel_size = kernel_size

        # Temporal conv + residual/skip 1x1s per layer
        self.tconvs = nn.ModuleList()
        self.gcns   = nn.ModuleList()
        self.res_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        c_res = residual_channels
        c_dil = dilation_channels
        for b in range(blocks):
            for l in range(layers_per_block):
                dilation = 2 ** l
                self.tconvs.append(ConvTemporalGLU(c_res, c_dil, kernel_size, dilation, dropout))
                # GCN mixing over supports will map back to residual_channels
                # We'll set supports dynamically in forward (A, A^T, [A_adp, A_adp^T])
                # Use 1x1 across concatenated supports
                # n_supports is 2 or 4 (static both dirs, + adaptive both dirs)
                # We'll create layer-wise mix conv lazily (we need n_supports) – simpler: always assume up to 4 and slice.
                self.gcns.append(nn.Conv2d(c_dil * 2, c_res, kernel_size=(1, 1)))  # placeholder; we’ll feed concat([xA, xA_T])
                self.res_convs.append(nn.Conv2d(c_res, c_res, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(c_dil, skip_channels, kernel_size=(1, 1)))

        # End (skip aggregation → output)
        self.end1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))

        # Adaptive adjacency parameters (node embeddings)
        if self.use_adp:
            self.E1 = nn.Parameter(torch.randn(n_nodes, adaptive_embed_dim) * (1.0 / math.sqrt(n_nodes)))
            self.E2 = nn.Parameter(torch.randn(n_nodes, adaptive_embed_dim) * (1.0 / math.sqrt(n_nodes)))
        else:
            self.register_parameter("E1", None)
            self.register_parameter("E2", None)

    def _make_adaptive_A(self) -> torch.Tensor:
        # Softmax(ReLU(E1 E2^T))
        logits = F.relu(self.E1 @ self.E2.T)  # [N,N]
        A = F.softmax(logits, dim=1)
        return A

    def forward(self, X_seq: torch.Tensor, A: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        """
        X_seq: [B, L, N, F]
        A:     [N, N]  (row-stochastic recommended; we also use A^T)
        """
        B, L, N, F_in = X_seq.shape
        assert N == self.n_nodes, f"n_nodes mismatch: dataset N={N}, model N={self.n_nodes}"
        # Reorder to [B, F, N, L]
        x = X_seq.permute(0, 3, 2, 1).contiguous()         # [B, F_in, N, L]
        x = self.start(x)                                   # [B, C_res, N, L]

        # A is 2-D here (we squeezed it in train loop), so a simple transpose is safe:
        if A.dim() == 3:      # [B,N,N] -> use the first (they're identical per batch)
            A = A[0]
        A = A.to(X_seq.dtype)            # <— add this line
        A_T = A.transpose(0, 1)
        supports: List[torch.Tensor] = [A, A_T]

        A_adp = None
        if self.use_adp:
            A_adp = self._make_adaptive_A()                # [N,N]
            supports.extend([A_adp, A_adp.T])

        skip = 0
        x_res = x
        idx = 0
        for b in range(self.blocks):
            for l in range(self.layers_per_block):
                # Temporal GLU on each node independently
                # reshape [B, C_res, N, L] -> merge N into batch for conv1d over time
                x_t = x_res.permute(0, 2, 1, 3).reshape(B * N, x_res.size(1), x_res.size(3))  # [B*N, C_res, L]
                x_t = self.tconvs[idx](x_t)  # [B*N, C_dil, L]
                c_dil = x_t.size(1)
                x_t = x_t.reshape(B, N, c_dil, x_t.size(-1)).permute(0, 2, 1, 3).contiguous()  # [B, C_dil, N, L]

                # Skip connection path (sum on time dim later)
                s = self.skip_convs[idx](x_t)   # [B, skip_channels, N, L]
                skip = s if isinstance(skip, int) else (skip + s)

                # Graph mixing over supports: xA, xA_T (and optionally adaptive)
                x_list = []
                for i, sup in enumerate(supports[:2]):  # always include A and A^T
                    x_list.append(self.nconv(x_t, sup))
                if A_adp is not None:
                    x_list.append(self.nconv(x_t, A_adp))
                    x_list.append(self.nconv(x_t, A_adp.T))

                # Concatenate along channel, then 1x1 back to residual_channels
                # We defined gcns with c_dil*2 output; if adaptive is on, we still feed only two tensors by summing adp into statics for simplicity.
                if A_adp is not None:
                    # simple blend: statics + adaptive (same size)
                    xa = x_list[0] + x_list[2]
                    xb = x_list[1] + x_list[3]
                    xg = torch.cat([xa, xb], dim=1)      # [B, 2*C_dil, N, L]
                else:
                    xg = torch.cat([x_list[0], x_list[1]], dim=1)  # [B, 2*C_dil, N, L]

                xg = self.gcns[idx](xg)                  # [B, C_res, N, L]

                # Residual 1x1 and add
                xg = self.res_convs[idx](xg)             # [B, C_res, N, L]
                x_res = x_res + xg

                idx += 1

        # Aggregate skip over time and project to output
        # Focus on the last time step (next-step prediction)
        # skip: [B, skip_channels, N, L]
        skip_last = skip[:, :, :, -1:]                   # [B, skip, N, 1]
        out = F.relu(self.end1(skip_last))
        out = self.end2(out)                             # [B, out_dim, N, 1]
        out = out.squeeze(-1).squeeze(1)                 # [B, N] (out_dim==1)
        return out
