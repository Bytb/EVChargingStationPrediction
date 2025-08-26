# strip_year_from_edgelist_static.py
import pandas as pd
import numpy as np
from pathlib import Path

def normalize_node(n):
    """Convert 'Road123_2019' -> 'Road123' by dropping the last '_' chunk if it looks like a year."""
    if pd.isna(n):
        return n
    s = str(n)
    parts = s.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) in (2, 4):
        return "_".join(parts[:-1])
    return parts[0] if len(parts) > 1 else s

def build_static_edgelist(in_csv, out_dir, src_col="source", dst_col="target", directed=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read
    df = pd.read_csv(in_csv, dtype=str)
    if src_col not in df.columns or dst_col not in df.columns:
        raise ValueError(f"Input must contain columns '{src_col}' and '{dst_col}'")

    # 2) Strip year from node names
    df['src_static'] = df[src_col].apply(normalize_node)
    df['dst_static'] = df[dst_col].apply(normalize_node)

    # 3) Drop self-loops
    df = df[df['src_static'] != df['dst_static']].copy()

    # 4) Deduplicate edges (treat as undirected unless directed=True)
    if not directed:
        a = np.minimum(df['src_static'], df['dst_static'])
        b = np.maximum(df['src_static'], df['dst_static'])
        df['u'] = a
        df['v'] = b
        df = df.drop_duplicates(['u', 'v'])
        edges_static = df[['u', 'v']].rename(columns={'u': 'source', 'v': 'target'})
    else:
        edges_static = df[['src_static', 'dst_static']].drop_duplicates() \
                         .rename(columns={'src_static': 'source', 'dst_static': 'target'})

    # 5) Write static string edge list
    edges_csv = out_dir / "edges_static.csv"
    edges_static.to_csv(edges_csv, index=False)

    # 6) Node index mapping
    nodes = pd.Index(edges_static['source']).union(edges_static['target'])
    nodes = pd.Series(nodes.sort_values(), name='node')
    nodes_df = nodes.reset_index(drop=True).to_frame()
    nodes_df['node_id'] = nodes_df.index.astype(int)
    nodes_csv = out_dir / "nodes_static.csv"
    nodes_df.to_csv(nodes_csv, index=False)

    # 7) Integer-indexed edges
    id_map = dict(zip(nodes_df['node'], nodes_df['node_id']))
    edges_idx = edges_static.copy()
    edges_idx['source_id'] = edges_idx['source'].map(id_map)
    edges_idx['target_id'] = edges_idx['target'].map(id_map)
    edges_idx_csv = out_dir / "edges_static_indexed.csv"
    edges_idx[['source_id', 'target_id']].to_csv(edges_idx_csv, index=False)

    print(f"Wrote:\n  {edges_csv}\n  {nodes_csv}\n  {edges_idx_csv}")

def main():
    path = 'tampa'
    # ---- HARD-CODED CONFIG ----
    in_csv = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\edges_fullscale.csv"
    out_dir = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\edges_RNN.csv"
    src_col = "source"   # change if your column names differ
    dst_col = "target"
    directed = False     # set True if your graph is directed
    # ---------------------------

    build_static_edgelist(in_csv, out_dir, src_col, dst_col, directed)

if __name__ == "__main__":
    main()
