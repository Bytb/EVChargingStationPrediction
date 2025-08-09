import arcpy
import os
from tqdm import tqdm

# === USER INPUTS ===
# C:\Users\Caleb\AppData\Local\Programs\ArcGIS\Pro\bin\Python\envs\arcgispro-py3
input_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Merged.shpp"  # <-- Replace with your input shapefile
output_path = r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\ArcGISData\RoadProcessingTestCode\NHS_Segments.shp"                 # <-- Final output (split lines)

# === Allow overwrite ===
arcpy.env.overwriteOutput = True

# === RUN FEATURE TO LINE ===
print("📐 Running Feature To Line...")
arcpy.management.FeatureToLine(
    in_features=input_path,
    out_feature_class=output_path,
    cluster_tolerance=""
)

print("✅ Done! Split lines saved to:", output_path)
