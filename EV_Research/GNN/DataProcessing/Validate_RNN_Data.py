# validate_rnn_artifacts.py
import numpy as np
import pandas as pd
from pathlib import Path

def validate_graph(edges_idx_csv, nodes_csv):
    print("\n[GRAPH]")
    edges = pd.read_csv(edges_idx_csv)
    nodes = pd.read_csv(nodes_csv)
    assert {"source_id","target_id"} <= set(edges.columns)
    assert {"node","node_id"} <= set(nodes.columns)

    n = len(nodes)
    assert edges[["source_id","target_id"]].min().min() >= 0
    assert edges[["source_id","target_id"]].max().max() < n

    # no self loops (expected for road touches)
    self_loops = (edges["source_id"] == edges["target_id"]).sum()
    print(f" nodes: {n:,} | edges: {len(edges):,} | self_loops: {self_loops}")
    assert self_loops == 0, "Unexpected self-loops present."

    # undirected dedup check
    canon = edges.assign(a=np.minimum(edges.source_id, edges.target_id),
                         b=np.maximum(edges.source_id, edges.target_id))
    dup = canon.duplicated(["a","b"]).sum()
    print(f" undirected duplicate edges: {dup}")
    assert dup == 0, "Undirected duplicates found (should have been deduped)."

def validate_features(x_path, mask_path, road_ids_csv, years_csv, feature_names_txt):
    print("\n[FEATURES]")
    X = np.load(x_path)           # (T,N,F)
    M = np.load(mask_path)        # (T,N)
    roads = pd.read_csv(road_ids_csv).sort_values("node_id")
    years = pd.read_csv(years_csv).sort_values("t_idx")
    feat_names = Path(feature_names_txt).read_text().strip().splitlines()
    T, N, F = X.shape
    print(f" X shape: (T,N,F) = {X.shape} | mask: {M.shape}")
    print(f" years: {years['year'].tolist()[:3]} ... {years['year'].tolist()[-3:]}")
    print(f" roads N={len(roads)} | features F={len(feat_names)}")
    assert M.shape == (T,N)
    assert len(roads) == N
    assert len(years) == T
    assert len(feat_names) == F

    # mask vs NaN consistency
    nan_locs = np.isnan(X).all(axis=2)  # True if ALL features NaN for (t,n)
    bad_mask = (M == False) & (~nan_locs)
    if bad_mask.any():
        t, n = np.argwhere(bad_mask)[0]
        raise AssertionError(f"Mask says missing but X has values at (t={t}, n={n})")

    # quick stats
    filled = M.sum()
    print(f" filled feature rows: {filled:,} / {T*N:,} ({filled/(T*N):.1%})")

def validate_labels(y_path, ymask_path, road_ids_csv, years_csv, horizon_drop=None):
    print("\n[LABELS]")
    y = np.load(y_path)           # (T,N) or (T-h,N) depending on your builder
    Ym = np.load(ymask_path)      # (T,N) or (T-h,N)
    roads = pd.read_csv(road_ids_csv).sort_values("node_id")
    years = pd.read_csv(years_csv).sort_values("t_idx")
    T_feat, N = len(years), len(roads)
    T_lab = y.shape[0]
    print(f" y shape: {y.shape} | y_mask: {Ym.shape}")

    # Allow labels to be shorter in time if you dropped last h years
    if T_lab == T_feat:
        pass
    elif T_lab < T_feat:
        print(f" labels have fewer timesteps than features (likely due to forecast horizon). "
              f"Features T={T_feat}, Labels T={T_lab}")
    else:
        raise AssertionError("Labels have MORE timesteps than features.")

    assert y.shape[1] == N == Ym.shape[1]
    assert Ym.shape == y.shape

    # mask vs NaN consistency
    nan_ok = np.isnan(y)
    bad = (~nan_ok) & (~Ym)   # value present but mask=False
    if bad.any():
        t, n = np.argwhere(bad)[0]
        raise AssertionError(f"Label present but mask=False at (t={t}, n={n})")

    # horizon sanity (optional): final H rows mostly False if you used 2-year growth
    if horizon_drop is not None and T_lab == T_feat:
        tail_true = Ym[-horizon_drop:].sum()
        print(f" last {horizon_drop} years labeled count: {tail_true}")
        # not asserting—just informative

    print(f" labeled cells: {Ym.sum():,} / {y.size:,} ({Ym.sum()/y.size:.1%})")

def cross_consistency(nodes_csv, edges_idx_csv, road_ids_csv):
    print("\n[CROSS-CONSISTENCY]")
    nodes_edges = pd.read_csv(nodes_csv).sort_values("node_id")["node"].tolist()
    nodes_feats = pd.read_csv(road_ids_csv).sort_values("node_id")["road_id"].tolist()
    assert nodes_edges == nodes_feats, \
        "Node orders differ between edges 'nodes_static.csv' and features 'road_ids.csv'."
    print(" node order consistent between edges and features ✔")

    # check that all nodes in edges appear in features (IDs exist)
    edges = pd.read_csv(edges_idx_csv)
    max_id = edges[["source_id","target_id"]].max().max()
    assert max_id < len(nodes_feats), "Edge node_id out of range."
    print(" all edge node_ids are within features N ✔")


if __name__ == "__main__":
    # ==== HARD-CODED PATHS (edit for your setup) ====
    # paths
    path = 'tampa'
    # Features
    x_path        = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\X.npy"
    mask_path     = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\mask.npy"
    road_ids_csv  = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\road_ids.csv"
    years_csv     = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\years.csv"
    feat_names    = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\feature_names.txt"

    # Labels
    y_path        = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\labels_RNN\\y.npy"
    ymask_path    = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\labels_RNN\\y_mask.npy"
    
    # Edges
    edges_static = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\edges_RNN\\edges_static.csv"    # string names (source,target)

    E  = pd.read_csv(edges_static)
    FN = pd.read_csv(road_ids_csv).sort_values("node_id")  # enforce same order as X.npy
    id_map = dict(zip(FN["road_id"].astype(str), FN["node_id"]))

    Ei = E.copy()
    Ei["source_id"] = Ei["source"].astype(str).map(id_map)
    Ei["target_id"] = Ei["target"].astype(str).map(id_map)

    out = Path("edges_RNN_synced"); out.mkdir(parents=True, exist_ok=True)
    Ei[["source_id","target_id"]].to_csv(out/"edges_static_indexed.csv", index=False)
    FN.rename(columns={"road_id":"node"}, inplace=True)
    FN[["node","node_id"]].to_csv(out/"nodes_static.csv", index=False)
    E.to_csv(out/"edges_static.csv", index=False)
    print("Wrote edges_RNN_synced/")

    edges_idx_csv = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\edges_RNN_synced\\edges_static_indexed.csv"
    nodes_csv     = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\edges_RNN_synced\\nodes_static.csv"
    
    validate_graph(edges_idx_csv, nodes_csv)
    validate_features(x_path, mask_path, road_ids_csv, years_csv, feat_names)
    validate_labels(y_path, ymask_path, road_ids_csv, years_csv, horizon_drop=2)
    cross_consistency(nodes_csv, edges_idx_csv, road_ids_csv)
    print("\nAll basic checks passed ✔")

