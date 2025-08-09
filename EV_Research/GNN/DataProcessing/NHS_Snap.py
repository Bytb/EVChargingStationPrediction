import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
line_file = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Segments.shp"
snap_distance = 3  # meters
output_file = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Snap.shp"

# --- Load and Project ---
print("📂 Loading and projecting data...")
gdf = gpd.read_file(line_file)
gdf = gdf.to_crs(epsg=5070)  # Make sure it's in projected CRS (meters)

# --- Step 1: Extract Endpoints and Line Index ---
print("📌 Extracting endpoints...")
line_indices = []
endpoints = []

for i, geom in enumerate(tqdm(gdf.geometry, desc="Extracting")):
    coords = list(geom.coords)
    endpoints.append(Point(coords[0]))
    endpoints.append(Point(coords[-1]))
    line_indices.extend([i, i])

endpoint_gdf = gpd.GeoDataFrame({'line_idx': line_indices, 'geometry': endpoints}, crs=gdf.crs)

# --- Step 2: Identify Connected Endpoints via STRtree ---
print("🔗 Building spatial index for connections...")
str_tree = STRtree(list(gdf.geometry))
connected_mask = np.zeros(len(endpoint_gdf), dtype=bool)

for i, pt in enumerate(tqdm(endpoint_gdf.geometry, desc="Checking connections")):
    if not isinstance(pt, Point):
        continue  # skip if somehow not a point

    try:
        candidates = str_tree.query(pt.buffer(0.01))
    except Exception:
        continue

    for geom in candidates:
        if geom is None or not hasattr(geom, "intersects"):
            continue
        try:
            if pt.intersects(geom):
                coords = list(geom.coords)
                if not (pt.equals(Point(coords[0])) or pt.equals(Point(coords[-1]))):
                    connected_mask[i] = True
                    break
        except Exception:
            continue

# --- Step 3: Filter Unconnected Endpoints ---
print("🧹 Filtering unconnected endpoints...")
unconnected = endpoint_gdf.loc[~connected_mask].reset_index(drop=True)

if len(unconnected) == 0:
    print("✅ No unconnected endpoints found. Skipping snapping.")
    gdf.to_file(output_file)
    exit()

# --- Step 4: Snap Unconnected Endpoints Using KDTree ---
print("📐 Snapping unconnected endpoints...")
coords_array = np.array([[pt.x, pt.y] for pt in unconnected.geometry])
tree = cKDTree(coords_array)
snap_targets = coords_array.copy()

for i in tqdm(range(len(coords_array)), desc="Snapping"):
    neighbors = tree.query_ball_point(coords_array[i], r=snap_distance)
    neighbors = [n for n in neighbors if n != i]
    if neighbors:
        # Snap to the closest neighbor
        dists = np.linalg.norm(coords_array[neighbors] - coords_array[i], axis=1)
        nearest_idx = neighbors[np.argmin(dists)]
        snap_targets[i] = coords_array[nearest_idx]

# --- Step 5: Update Line Geometries ---
print("🛠️ Replacing line endpoints...")
lines_updated = []
unconnected_idx_map = dict(zip(zip(unconnected.line_idx, unconnected.geometry.apply(lambda p: (p.x, p.y))), snap_targets))

for i, geom in tqdm(enumerate(gdf.geometry), total=len(gdf), desc="Updating lines"):
    coords = list(geom.coords)
    for j in [0, -1]:
        key = (i, (coords[j][0], coords[j][1]))
        if key in unconnected_idx_map:
            coords[j] = tuple(unconnected_idx_map[key])
    lines_updated.append(LineString(coords))

gdf.geometry = lines_updated

# --- Save Output ---
print("💾 Saving output...")
gdf.to_file(output_file)
print(f"✅ Finished! Snapped lines saved to: {output_file}")
