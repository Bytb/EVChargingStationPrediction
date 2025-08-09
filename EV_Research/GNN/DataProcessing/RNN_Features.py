# build_rnn_features.py
import pandas as pd
import numpy as np
from pathlib import Path

def parse_road_year(roadyear: str):
    s = str(roadyear)
    parts = s.split("_")
    if len(parts) < 2:
        raise ValueError(f"Bad RoadYearID: {roadyear}")
    road = "_".join(parts[:-1])
    year = int(parts[-1])
    return road, year

def build_rnn_features(
    in_csv: str,
    out_dir: str,
    id_col: str = "RoadYearID",
    nodes_static_csv: str = None,  # optional, to lock node order to your edge list nodes
    drop_cols: list = None,        # optional: columns to exclude from features
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv)
    if id_col not in df.columns:
        raise ValueError(f"Expected id_col '{id_col}' in CSV")

    # Parse RoadID + Year from RoadYearID
    roads, years = zip(*df[id_col].map(parse_road_year))
    df["RoadID"] = roads
    df["Year"] = years

    # Choose feature columns (everything except IDs/Year + optional drops)
    base_drop = {id_col, "RoadID", "Year"}
    if drop_cols:
        base_drop |= set(drop_cols)
    feature_cols = [c for c in df.columns if c not in base_drop]

    # Time and node ordering
    unique_years = sorted(df["Year"].unique().tolist())
    if nodes_static_csv and Path(nodes_static_csv).exists():
        nodes = pd.read_csv(nodes_static_csv)
        if {"node","node_id"}.issubset(nodes.columns):
            road_order = nodes.sort_values("node_id")["node"].tolist()
        else:
            road_order = sorted(df["RoadID"].unique().tolist())
    else:
        road_order = sorted(df["RoadID"].unique().tolist())

    T, N, F = len(unique_years), len(road_order), len(feature_cols)
    X = np.full((T, N, F), np.nan, dtype=np.float64)
    mask = np.zeros((T, N), dtype=bool)

    year_to_idx = {y:i for i,y in enumerate(unique_years)}
    road_to_idx = {r:i for i,r in enumerate(road_order)}

    # Ensure numeric
    df_feat = df[["RoadID","Year"] + feature_cols].copy()
    for col in feature_cols:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

    # Fill tensor
    for _, row in df_feat.iterrows():
        t = year_to_idx[row["Year"]]
        n = road_to_idx.get(row["RoadID"])
        if n is None:
            continue
        X[t, n, :] = row[feature_cols].values
        mask[t, n] = True

    # Save artifacts
    np.save(out/"X.npy", X)
    np.save(out/"mask.npy", mask)
    pd.DataFrame({"road_id": road_order, "node_id": range(N)}).to_csv(out/"road_ids.csv", index=False)
    pd.DataFrame({"year": unique_years, "t_idx": range(T)}).to_csv(out/"years.csv", index=False)
    Path(out/"feature_names.txt").write_text("\n".join(feature_cols))

    print(f"Saved X.npy with shape (T,N,F) = {X.shape}")
    print(f"Filled slots: {int(mask.sum())} / {mask.size} | Missing: {int(mask.size - mask.sum())}")
    print(f"Features ({F}): {feature_cols}")

if __name__ == "__main__":
    # ---- HARD-CODED CONFIG ----
    path = 'tampa'
    in_csv = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\Final_Joined_Features.csv"
    out_dir = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN"
    nodes_static_csv = None  # e.g., r"C:\path\to\nodes_static.csv" to lock node order
    drop_cols = []           # e.g., ["SomeNonFeatureColumn"]
    # ---------------------------
    build_rnn_features(in_csv, out_dir, id_col="RoadYearID", nodes_static_csv=nodes_static_csv, drop_cols=drop_cols)
