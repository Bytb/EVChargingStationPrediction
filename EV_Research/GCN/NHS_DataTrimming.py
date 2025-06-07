import geopandas as gpd

# Define file paths
input_filepath = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\National_Highway_System_Test\TampaRoads.shp"
output_filepath = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\National_Highway_System_Test\Filtered_Data\Filtered_NHS.shp"

# Read the original shapefile
gdf = gpd.read_file(input_filepath)
print("Original Shape:", gdf.shape)

# Keep only 'ID' and 'geometry' columns, and filter first 25,000 rows
gdf = gdf[['ID', 'geometry']]
print("Filtered Shape:", gdf.shape)

# Export the filtered GeoDataFrame to a new shapefile
gdf.to_file(output_filepath, driver="ESRI Shapefile")

print(f"Filtered shapefile saved to: {output_filepath}")
