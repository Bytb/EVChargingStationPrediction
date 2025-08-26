# graph/loader.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from datetime import datetime


NormKind = Literal["gcn", "rw"]  # gcn = D^{-1/2}(A+I)D^{-1/2}, rw = D^{-1}A (optionally +I)


def _to_coo(rows: np.ndarray, cols: np.ndarray, N: int) -> sp.coo_matrix:
    """Build undirected, unweighted COO adjacency (duplicates coalesced)."""
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    # Coalesce duplicates by summing then clamping to 1
    A = A.tocsr()
    A.data[:] = 1.0
    return A.tocoo()


def _symmetrize(A: sp.coo_matrix) -> sp.coo_matrix:
    return (A + A.T).tocoo()


def _drop_self_loops(A: sp.coo_matrix) -> sp.coo_matrix:
    A = A.tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocoo()


def _add_self_loops(A: sp.csr_matrix) -> sp.csr_matrix:
    I = sp.eye(A.shape[0], dtype=A.dtype, format="csr")
    return (A + I).tocsr()


def _gcn_normalize(A: sp.csr_matrix, add_self_loops: bool = True) -> sp.csr_matrix:
    """D^{-1/2} (A [+ I]) D^{-1/2} (symmetric)."""
    if add_self_loops:
        A = _add_self_loops(A)
    deg = np.asarray(A.sum(axis=1)).ravel()
    # avoid divide-by-zero
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt.astype(np.float32))
    return (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def _row_normalize(A: sp.csr_matrix, add_self_loops: bool = False) -> sp.csr_matrix:
    """Row-stochastic D^{-1} (A [+ I]) (random-walk)."""
    if add_self_loops:
        A = _add_self_loops(A)
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=np.float32)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    D_inv = sp.diags(inv)
    return (D_inv @ A).tocsr()


@dataclass
class GraphMeta:
    N: int
    E_undirected: int
    density: float
    isolates: int
    isolates_examples: list
    degree_min: int
    degree_mean: float
    degree_median: float
    degree_p95: float
    degree_max: int
    n_components: int
    largest_component: int
    created_at: str


class GraphLoader:
    """
    Loads static road graph and emits normalized adjacencies.

    Inputs:
      - nodes_static.csv (must contain 'node_id' and optionally 'node')
      - edges_static_indexed.csv (must contain 'source_id','target_id' OR 'source','target')

    Artifacts:
      - out_dir/A_raw.npz  (undirected, no self-loops)
      - out_dir/graph_meta.json
      - out_dir/A_{kind}_sl{0|1}.npz on first request (cached)
    """

    def __init__(self, nodes_csv: str | Path, edges_csv: str | Path, out_dir: str | Path):
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Loaded on demand
        self.N: Optional[int] = None
        self.A_raw: Optional[sp.csr_matrix] = None
        self.meta: Optional[Dict] = None

    # -------------------- LOADERS --------------------

    def load_raw(self, force_reload: bool = False) -> sp.csr_matrix:
        if self.A_raw is not None and not force_reload:
            return self.A_raw

        nodes = pd.read_csv(self.nodes_csv)
        if "node_id" not in nodes.columns:
            raise ValueError("nodes_static.csv must contain a 'node_id' column.")
        N = int(nodes["node_id"].max()) + 1
        # optional sanity: ensure contiguous
        assert set(nodes["node_id"]) == set(range(N)), "node_id must be contiguous from 0..N-1"
        self.N = N

        edges = pd.read_csv(self.edges_csv)
        # accept either naming
        src_col = "source_id" if "source_id" in edges.columns else "source"
        dst_col = "target_id" if "target_id" in edges.columns else "target"
        if src_col not in edges.columns or dst_col not in edges.columns:
            raise ValueError("edges_static_indexed.csv must have columns 'source_id'/'target_id' or 'source'/'target'.")

        # drop self-loops, out-of-range, and duplicates (after making undirected)
        src = edges[src_col].astype(int).to_numpy(copy=False)
        dst = edges[dst_col].astype(int).to_numpy(copy=False)
        valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
        if valid.sum() < len(edges):
            print(f"[GraphLoader] Dropped {(~valid).sum()} edges with out-of-range endpoints.")
        src, dst = src[valid], dst[valid]

        # Make undirected and drop self-loops
        rows = np.concatenate([src, dst])
        cols = np.concatenate([dst, src])
        A = _to_coo(rows, cols, N)
        A = _drop_self_loops(A)

        # Convert to CSR for efficient math
        A = A.tocsr()

        # Save and summarize
        self.A_raw = A
        self.meta = self._summarize_and_save(A)
        sp.save_npz(self.out_dir / "A_raw.npz", A)
        return A

    def _summarize_and_save(self, A: sp.csr_matrix) -> Dict:
        deg = np.asarray(A.sum(axis=1)).ravel()
        isolates_idx = np.where(deg == 0)[0]
        n_comp, labels = connected_components(A, directed=False, return_labels=True)
        largest = int(np.bincount(labels).max())
        N = A.shape[0]
        E_undirected = int(A.nnz // 2)  # because undirected without self-loops

        meta = GraphMeta(
            N=N,
            E_undirected=E_undirected,
            density=(2 * E_undirected) / (N * (N - 1)) if N > 1 else 0.0,
            isolates=int(len(isolates_idx)),
            isolates_examples=[int(i) for i in isolates_idx[:10].tolist()],
            degree_min=int(deg.min(initial=0)),
            degree_mean=float(deg.mean() if N > 0 else 0.0),
            degree_median=float(np.median(deg) if N > 0 else 0.0),
            degree_p95=float(np.percentile(deg, 95) if N > 0 else 0.0),
            degree_max=int(deg.max(initial=0)),
            n_components=int(n_comp),
            largest_component=int(largest),
            created_at=datetime.utcnow().isoformat() + "Z",
        ).__dict__

        with open(self.out_dir / "graph_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[GraphLoader] N={meta['N']:,}, E(undirected)={meta['E_undirected']:,}, "
              f"isolates={meta['isolates']:,}, comps={meta['n_components']}, "
              f"largest_comp={meta['largest_component']:,}")
        return meta

    # -------------------- NORMALIZATION --------------------

    def get_adj(
        self,
        kind: NormKind = "gcn",
        add_self_loops: bool = True,
        cache: bool = True,
    ) -> sp.csr_matrix:
        """
        Returns normalized adjacency of requested kind.
        kind='gcn' → D^{-1/2}(A + I)D^{-1/2}
        kind='rw'  → D^{-1}A  (set add_self_loops=True if you want +I)
        """
        if self.A_raw is None:
            self.load_raw()

        cache_name = f"A_{kind}_sl{1 if add_self_loops else 0}.npz"
        cache_path = self.out_dir / cache_name
        if cache and cache_path.exists():
            return sp.load_npz(cache_path).tocsr()

        A = self.A_raw
        if kind == "gcn":
            A_norm = _gcn_normalize(A, add_self_loops=add_self_loops)
        elif kind == "rw":
            A_norm = _row_normalize(A, add_self_loops=add_self_loops)
        else:
            raise ValueError("kind must be 'gcn' or 'rw'.")

        if cache:
            sp.save_npz(cache_path, A_norm)
        return A_norm

    # -------------------- VERIFICATION --------------------

    def verify(
        self,
        features_x_npy: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> Dict[str, bool]:
        """
        Runs acceptance checks and returns a dict of booleans.
        Optionally pass features X.npy to assert N matches feature dim.
        """
        results = {}

        # Raw exists and symmetry (undirected no self-loops)
        A_raw = self.A_raw if self.A_raw is not None else sp.load_npz(self.out_dir / "A_raw.npz")
        results["raw_shape"] = (A_raw.shape[0] == A_raw.shape[1]) and (self.N is None or A_raw.shape[0] == self.N)
        results["raw_self_loops"] = A_raw.diagonal().sum() == 0
        # Symmetry: since it's CSR, compare to transpose sparsely
        diff = (A_raw - A_raw.T).nnz
        results["raw_symmetric"] = diff == 0

        # GCN normalized checks
        A_gcn = self.get_adj("gcn", add_self_loops=True, cache=False)
        diag_gcn = A_gcn.diagonal()
        results["gcn_diag_positive"] = np.all(diag_gcn > 0)  # self-loops ensure >0
        results["gcn_finite"] = np.isfinite(A_gcn.data).all()

        # Row-normalized checks (without self-loops by default)
        A_rw = self.get_adj("rw", add_self_loops=False, cache=False)
        row_sum = np.asarray(A_rw.sum(axis=1)).ravel()
        deg = np.asarray(A_raw.sum(axis=1)).ravel()
        # For non-isolates, row sums should be ~1
        nz = deg > 0
        results["rw_rowsum_ok"] = bool(np.allclose(row_sum[nz], 1.0, atol=1e-5))
        results["rw_finite"] = np.isfinite(A_rw.data).all()

        # Optional: check X.npy N matches
        if features_x_npy:
            x = np.load(features_x_npy, mmap_mode="r")
            # x shape [T, N, F]
            results["features_match_N"] = (A_raw.shape[0] == x.shape[1])
        else:
            results["features_match_N"] = True  # not checked

        # Report
        if verbose:
            print("\n[GraphLoader] Verification report")
            for k, v in results.items():
                print(f"  - {k}: {v}")
        return results


# -------------------- convenience function --------------------

def check_graph(out_dir: str | Path, features_x_npy: Optional[str | Path] = None) -> Dict[str, bool]:
    """
    Convenience: reloads A_raw and runs verification.
    Use this after the first run or anytime you want to re-check outputs in `out_dir`.
    """
    out_dir = Path(out_dir)
    # Load meta to discover node count; ensure A_raw exists
    A_path = out_dir / "A_raw.npz"
    if not A_path.exists():
        raise FileNotFoundError(f"No A_raw.npz in {out_dir}. Run GraphLoader.load_raw(...) first.")
    A_raw = sp.load_npz(A_path)  # noqa: F841
    # Build a dummy loader just for verify (no CSVs needed here)
    gl = GraphLoader(nodes_csv=out_dir / "_placeholder_nodes.csv",
                     edges_csv=out_dir / "_placeholder_edges.csv",
                     out_dir=out_dir)
    gl.A_raw = sp.load_npz(out_dir / "A_raw.npz")
    gl.N = gl.A_raw.shape[0]
    return gl.verify(features_x_npy=features_x_npy, verbose=True)


# -------------------- example usage (comment out in production) --------------------
# if __name__ == "__main__":
#     loader = GraphLoader(
#         nodes_csv="data/nodes_static.csv",
#         edges_csv="data/edges_static_indexed.csv",
#         out_dir="artifacts/graph",
#     )
#     loader.load_raw()
#     # Build and cache both normalizations
#     _ = loader.get_adj("gcn", add_self_loops=True)
#     _ = loader.get_adj("rw", add_self_loops=False)
#     # Run full verification (optionally pass your X.npy path)
#     loader.verify(features_x_npy=None, verbose=True)
#     # Later, you can call:
#     # check_graph("artifacts/graph", features_x_npy="features_RNN/X.npy")
