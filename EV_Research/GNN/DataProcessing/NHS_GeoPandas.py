import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring
import numpy as np
from tqdm import tqdm

# === USER INPUTS ===
input_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Test.shp"         # e.g., "National_NHS_20240501.shp"
output_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Merged.shp"       # e.g., "Roads.shp"
working_crs = "EPSG:5070"                    # NAD83 / Conus Albers (meters)
EXTEND_DIST = 500             # Per-end = 500m (1 km total trim)

# --- Extend and trim each segment individually ---
def extend_and_trim(geom, distance):
    if isinstance(geom, MultiLineString):
        parts = [extend_and_trim(part, distance) for part in geom.geoms]
        return MultiLineString([g for g in parts if g is not None])

    if not isinstance(geom, LineString) or len(geom.coords) < 2:
        return None

    coords = list(geom.coords)

    # Extend start
    p1, p2 = coords[0], coords[1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    len1 = np.hypot(dx, dy)
    x1 = p1[0] - distance * dx / len1
    y1 = p1[1] - distance * dy / len1

    # Extend end
    p3, p4 = coords[-2], coords[-1]
    dx, dy = p4[0] - p3[0], p4[1] - p3[1]
    len2 = np.hypot(dx, dy)
    x2 = p4[0] + distance * dx / len2
    y2 = p4[1] + distance * dy / len2

    extended = LineString([(x1, y1)] + coords[1:-1] + [(x2, y2)])

    # Only trim if long enough
    if extended.length > 2 * distance:
        return substring(extended, distance, extended.length - distance)
    else:
        return extended
    # return extended

# --- Step 1: Load and reproject ---
print("📥 Loading and projecting...")
gdf = gpd.read_file(input_path)
original_crs = gdf.crs
gdf = gdf.to_crs(working_crs)

# --- Step 2: Extend and trim per feature ---
print("🔁 Extending + trimming each segment...")
gdf["geometry"] = [extend_and_trim(geom, EXTEND_DIST) for geom in tqdm(gdf.geometry, desc="Processing")]

# Drop any None rows (invalid or too-short)
gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)

# --- NEW: Dissolve entire dataset into one geometry ---
print("🧪 Dissolving into one geometry...")
gdf = gpd.GeoDataFrame(geometry=[gdf.geometry.union_all()], crs=gdf.crs)

# --- Step 3: Restore CRS and save ---
print("🌐 Reprojecting and saving...")
if original_crs:
    gdf = gdf.to_crs(original_crs)

gdf.to_file(output_path)
print("✅ Done! Output saved to:", output_path)

# import geopandas as gpd
# from shapely.geometry import LineString, MultiLineString
# from shapely.ops import substring
# import numpy as np
# from tqdm import tqdm

# # === USER INPUTS ===
# input_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\FullScale\National_highway_system_FHWA\NHS_RoadID.shp"
# output_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\FullScale\National_highway_system_FHWA\NHS_Extended.shp"
# working_crs = "EPSG:5070"  # NAD83 / Conus Albers (meters)
# EXTEND_DIST = 500  # Extend each end by 500m = 1km total

# # --- Extend a single linestring ---
# def extend_line(geom, distance):
#     if isinstance(geom, MultiLineString):
#         parts = [extend_line(part, distance) for part in geom.geoms]
#         return MultiLineString([g for g in parts if g is not None])

#     if not isinstance(geom, LineString) or len(geom.coords) < 2:
#         return None

#     coords = list(geom.coords)

#     # Extend start
#     p1, p2 = coords[0], coords[1]
#     dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
#     len1 = np.hypot(dx1, dy1)
#     x1 = p1[0] - distance * dx1 / len1
#     y1 = p1[1] - distance * dy1 / len1

#     # Extend end
#     p3, p4 = coords[-2], coords[-1]
#     dx2, dy2 = p4[0] - p3[0], p4[1] - p3[1]
#     len2 = np.hypot(dx2, dy2)
#     x2 = p4[0] + distance * dx2 / len2
#     y2 = p4[1] + distance * dy2 / len2

#     return LineString([(x1, y1)] + coords[1:-1] + [(x2, y2)])

# # --- Trim a single linestring ---
# def trim_line(geom, distance):
#     if not isinstance(geom, LineString) or geom.length <= 2 * distance:
#         return None
#     return substring(geom, distance, geom.length - distance)

# # --- Step 1: Load and reproject ---
# print("📥 Loading and projecting...")
# gdf = gpd.read_file(input_path)
# original_crs = gdf.crs
# gdf = gdf.to_crs(working_crs)

# # --- Step 2: Extend each line ---
# print("📏 Extending lines...")
# gdf["geometry"] = [extend_line(geom, EXTEND_DIST) for geom in tqdm(gdf.geometry, desc="Extending")]
# gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)

# # --- Step 3: Merge all geometries ---
# print("🧬 Merging geometries into one...")
# merged = gdf.geometry.union_all()

# # --- Step 4: Explode merged geometry back into segments ---
# print("🔀 Exploding merged geometry...")
# if isinstance(merged, MultiLineString):
#     segments = list(merged.geoms)
# else:
#     segments = [merged]
# gdf = gpd.GeoDataFrame(geometry=segments, crs=gdf.crs)

# # --- Step 5: Trim each segment back ---
# print("✂️ Trimming back extensions...")
# gdf["geometry"] = [trim_line(geom, EXTEND_DIST) for geom in tqdm(gdf.geometry, desc="Trimming")]
# gdf = gdf[gdf.geometry.notnull()].reset_index(drop=True)

# # --- Step 6: Restore CRS and save ---
# print("🌐 Reprojecting and saving...")
# if original_crs:
#     gdf = gdf.to_crs(original_crs)

# gdf.to_file(output_path)
# print("✅ Done! Output saved to:", output_path)
