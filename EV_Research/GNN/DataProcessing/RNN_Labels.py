# build_rnn_labels.py
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

def build_rnn_labels(
    labels_csv: str,
    out_dir: str,
    road_ids_csv: str,   # from features step (road_id,node_id)
    years_csv: str,      # from features step (year,t_idx)
    id_col: str = None,  # e.g., "RoadYearID" if present; if None, expects RoadID + Year columns
    label_col: str = None,  # override if your label column has a specific name
    agg: str = "mean",   # how to resolve duplicates (mean|sum|first)
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # --- load mapping for consistent ordering ---
    roads_df = pd.read_csv(road_ids_csv)
    years_df = pd.read_csv(years_csv)
    if not {"road_id","node_id"} <= set(roads_df.columns):
        raise ValueError("road_ids_csv must have columns: road_id,node_id")
    if not {"year","t_idx"} <= set(years_df.columns):
        raise ValueError("years_csv must have columns: year,t_idx")

    road_order = roads_df.sort_values("node_id")["road_id"].tolist()
    year_order = years_df.sort_values("t_idx")["year"].tolist()
    road_to_idx = {r:i for i,r in enumerate(road_order)}
    year_to_idx = {y:i for i,y in enumerate(year_order)}

    # --- read labels ---
    df = pd.read_csv(labels_csv)

    # detect label column if not provided
    if label_col is None:
        candidates = [c for c in ["label","Label","y","target","Target"] if c in df.columns]
        if not candidates:
            # fallback: last numeric column
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError("Could not find a numeric label column; set label_col explicitly.")
            label_col = num_cols[-1]
        else:
            label_col = candidates[0]

    # normalize RoadID + Year
    if id_col and id_col in df.columns:
        roads, years = zip(*df[id_col].map(parse_road_year))
        df["RoadID"] = roads
        df["Year"] = years
    else:
        if not {"RoadID","Year"} <= set(df.columns):
            raise ValueError("Expected either id_col present or columns RoadID and Year.")
        # ensure types
        df["RoadID"] = df["RoadID"].astype(str)
        df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)

    # keep only needed cols
    df = df[["RoadID","Year",label_col]].copy()
    df.rename(columns={label_col: "label"}, inplace=True)

    # coerce numeric labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # filter to roads/years we know about (so shapes match X.npy)
    df = df[df["RoadID"].isin(road_to_idx.keys()) & df["Year"].isin(year_to_idx.keys())]

    # resolve duplicates if any
    if agg == "mean":
        df = df.groupby(["RoadID","Year"], as_index=False)["label"].mean()
    elif agg == "sum":
        df = df.groupby(["RoadID","Year"], as_index=False)["label"].sum()
    elif agg == "first":
        df = df.drop_duplicates(["RoadID","Year"], keep="first")
    else:
        raise ValueError("agg must be one of: mean|sum|first")

    # --- build (T,N) tensor and mask ---
    T, N = len(year_order), len(road_order)
    y = np.full((T, N), np.nan, dtype=np.float64)
    ymask = np.zeros((T, N), dtype=bool)

    for _, row in df.iterrows():
        t = year_to_idx[row["Year"]]
        n = road_to_idx[row["RoadID"]]
        y[t, n] = row["label"]
        ymask[t, n] = True

    # save arrays
    np.save(out/"y.npy", y)
    np.save(out/"y_mask.npy", ymask)

    # also save a human‑readable matrix (years as rows, roads as columns)
    wide = pd.DataFrame(index=year_order, columns=road_order, data=y)
    wide.index.name = "year"
    wide.to_csv(out/"labels_matrix.csv")  # NaN where missing

    # and a tidy long form
    long = df.copy()
    long = long.sort_values(["Year","RoadID"])
    long.to_csv(out/"labels_long.csv", index=False)

    print(f"Saved y.npy (T,N) = {y.shape}, filled {ymask.sum()} / {ymask.size}")
    print(f"Wrote: {out/'y.npy'}, {out/'y_mask.npy'}, {out/'labels_matrix.csv'}, {out/'labels_long.csv'}")
    print("Label column used:", label_col)

if __name__ == "__main__":
    # ---- HARD-CODED CONFIG ----
    path = 'tampa'
    labels_csv     = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\Labels_Test.csv"
    out_dir        = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\labels_RNN"
    road_ids_csv   = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\road_ids.csv"  # from the features step
    years_csv      = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{path}\\features_RNN\\years.csv"     # from the features step
    id_col         = "RoadYear"   # set to None if your labels already have RoadID + Year columns
    label_col      = None           # set if you know the exact column name
    agg            = "mean"         # how to combine duplicates
    # ---------------------------
    build_rnn_labels(labels_csv, out_dir, road_ids_csv, years_csv, id_col, label_col, agg)
