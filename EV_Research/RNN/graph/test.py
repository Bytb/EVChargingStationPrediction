# RNN/graph/test.py
from pathlib import Path
import yaml

# If you kept `from graph.loader import GraphLoader`, ensure packages or set PYTHONPATH=.
# If you switch to local import, uncomment this:
from loader import GraphLoader

def load_cfg(yaml_path: Path) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # project root = repo root (test.py is RNN/graph/test.py → parents[2])
    ROOT = Path(__file__).resolve().parents[3]
    CFG_PATH = ROOT / "EV_Research" / "RNN" / "config" / "default.yaml"
    cfg = load_cfg(CFG_PATH)

    city = cfg["dataset"]["city"]
    city_root = ROOT / "data" / city

    features_dir = city_root / "features_RNN"
    labels_dir   = city_root / "labels_RNN"       # not used here, but for later
    edges_dir    = city_root / "edges_RNN_synced"
    graph_dir    = city_root / "graph_static"

    # ---- Step 2: build/verify static graph for this city ----
    gl = GraphLoader(
        nodes_csv=edges_dir / "nodes_static.csv",
        edges_csv=edges_dir / "edges_static_indexed.csv",
        out_dir=graph_dir,
    )

    if cfg["graph"]["build_if_missing"] or not (graph_dir / "A_raw.npz").exists():
        gl.load_raw()
        _ = gl.get_adj(
            cfg["graph"]["norm"],
            add_self_loops=cfg["graph"]["add_self_loops"]
        )

    if cfg["graph"]["verify_with_features"]:
        gl.verify(features_x_npy=features_dir / "X.npy", verbose=True)
