import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm

TEST = 'TestRun(Tampa)'
# ---- CONFIG ----
road_path = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\ArcGISData\\{TEST}\\Segments\\Roads.shp"
station_folder = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\ArcGISData\\{TEST}\\Labels\\FilteredStationLocationsPerYear"
output_csv = f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\ArcGISData\\{TEST}\\AvgDistance_StationCount\\StationCounts_AllYears.csv"
buffer_dist_meters = 8046.72  # 5 statute miles in meters
years = range(2016, 2024)

# ---- LOAD ROADS AND PREP ----
roads = gpd.read_file(road_path).to_crs(epsg=5070)
#roads["RoadID"] = roads["RoadID"].astype(int) + 1

# Buffer roads
roads_buffered = roads.copy()
roads_buffered["geometry"] = roads_buffered.geometry.buffer(buffer_dist_meters)
roads_buffered = roads_buffered.explode(index_parts=False)

# ---- LOOP OVER YEARS ----
all_years = []

for year in tqdm(years):
    print(f"📅 Processing {year}...")
    station_path = os.path.join(station_folder, f"Stations_{year}.shp")

    if not os.path.exists(station_path):
        print(f"⚠️ Missing: {station_path}")
        continue

    stations = gpd.read_file(station_path).to_crs(roads_buffered.crs)

    # Basic geometry checks
    stations = stations[stations.geometry.notnull() & stations.geometry.is_valid]

    # Buffer each station slightly to avoid edge-miss issues (0.01 meters)
    stations["geometry"] = stations.buffer(0.01)

    # Explode multi-part geometries
    stations = stations.explode(index_parts=False)

    # Spatial join: include all stations that fall in any road buffer
    joined = gpd.sjoin(stations[["geometry"]], roads_buffered[["RoadID", "geometry"]], how="inner", predicate="within")

    # Count stations per road
    counts = joined.groupby("RoadID").size().reset_index(name="StationCount")

    # Merge with all roads to fill in 0s
    counts_full = pd.merge(roads[["RoadID"]], counts, on="RoadID", how="left").fillna(0)
    counts_full["StationCount"] = counts_full["StationCount"].astype(int)
    counts_full["RoadYearID"] = counts_full["RoadID"].apply(lambda x: f"Road{x}_{year}")

    all_years.append(counts_full[["RoadYearID", "StationCount"]])

# ---- EXPORT ----
final_df = pd.concat(all_years, ignore_index=True)
final_df.to_csv(output_csv, index=False)
print(f"✅ Exported: {output_csv}")
