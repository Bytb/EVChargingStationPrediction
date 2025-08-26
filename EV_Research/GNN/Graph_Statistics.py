import pandas as pd
import networkx as nx

# === USER INPUT ===
csv_path = r'C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\edges_fullscale.csv'  # Replace with your file path
source_col = 'source'  # Name of source column in the CSV
target_col = 'target'  # Name of target column in the CSV
is_directed = False     # Set True if your graph is directed

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# === BUILD GRAPH ===
if is_directed:
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, create_using=nx.DiGraph())
else:
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)

# === COMPUTE STATS ===
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)

# === PRINT RESULTS ===
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
print(f"Graph density: {density:.4f}")
