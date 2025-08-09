import networkx as nx
import geopandas as gpd
from tqdm import tqdm
from matplotlib import pyplot as plt

filepath = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\National_Highway_System_Test\Intersect_Tampa\Intersect_Tampa.shp"

gdf = gpd.read_file(filepath)

G = nx.Graph()

#Comparing all endpoints against each other:
for idx, row in tqdm(gdf.iterrows(), total = len(gdf), desc="Connecting Endpoints"):
    line = row.geometry
    for other_idx, other_row in gdf.iterrows():
        if idx >= other_idx: continue
        oline = other_row.geometry
        if line.touches(oline):
            G.add_edge(idx, other_idx)

def print_graph(G, title):
    print("Nodes: ", G.number_of_nodes())
    print("Edges: ", G.number_of_edges())

    # Increase plot size
    plt.figure(figsize=(18, 10))  # Adjust width and height as needed
    
    # # Get node positions (layout)
    pos = nx.spring_layout(G)

    # Use tqdm to track edge drawing progress
    print("Plotting edges...")
    for edge in tqdm(G.edges(), total=len(G.edges()), desc="Drawing edges"):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], alpha=0.5, edge_color="gray")

    # Draw nodes (after edges to ensure visibility)
    nx.draw_networkx_nodes(G, pos, node_size=3, node_color="blue")

    # Draw labels
    #nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Display the plot
    plt.title(title)
    plt.show()

print_graph(G, "Tampa Graph")