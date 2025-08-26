import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas as pd
import torch
import numpy as np

# Parameters
cities = ['tampa', 'seattle', 'dallas', 'philadelphia', 'atlanta', 'chicago', 'la']
all = ['tampa', 'seattle', 'dallas', 'philadelphia', 'atlanta', 'chicago', 'la', 'NHS']
test = ['tampa']
# Storage for all city skew data
skew_dfs = []

for city in test:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST = city
    SPATIO_TEMPORAL = True

    #edges = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Raw_Edges_Expanded.csv")
    if SPATIO_TEMPORAL:
        edges = pd.read_csv(f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{TEST}\\edges_fullscale_with_time.csv")
    else:
        edges = pd.read_csv(f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{TEST}\\edges_fullscale.csv")
    features = pd.read_csv(f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{TEST}\\Final_Joined_Features.csv", index_col=0)
    labels = pd.read_csv(f"C:\\Users\\Caleb\\OneDrive - University of South Florida\\EV_Research\\EV_Research_PythonCode\\data\\{TEST}\\Labels_Test.csv", index_col=0)

    # --- Format IDs ---
    def adjust_edge(entry):
        road, year = entry.replace("Road", "").split("_")
        return f"Road{int(road)}_{year}"

    edges['source'] = edges['source'].apply(adjust_edge)
    edges['target'] = edges['target'].apply(adjust_edge)

    def adjust_road_id(entry):
        road, year = entry.replace("Road", "").split("_")
        return f"Road{int(road)}_{year}"

    features.index = features.index.to_series().apply(adjust_road_id)

    # --- Testing Skew ---
    #numeric_cols = features.select_dtypes(include=[np.number]).columns

    # Compute skew
    skew_stats = features.skew().sort_index()  # keep same column order
    skew_df = pd.DataFrame(skew_stats, columns=[city])
    skew_dfs.append(skew_df)

# Combine into one table
skew_table = pd.concat(skew_dfs, axis=1).round(2)

# Function for highlight
def highlight_high_skew(val):
    color = 'background-color: yellow' if abs(val) > 1.0 else ''
    return color

# Display with highlighting (Jupyter or other rich display)
try:
    from IPython.display import display
    display(skew_table.style.applymap(highlight_high_skew))
except ImportError:
    print(skew_table)

# Save plain table to CSV
skew_table.to_csv("city_skew_statistics.csv")

print("\n📊 Skew table saved to city_skew_statistics.csv")

# --- Normalize Features ---
# def normalize_features(features, log_transform_cols=None, future_year=None):
#     feature_cols = features.columns.tolist()
#     if log_transform_cols:
#         for col in log_transform_cols:
#             if col in features.columns:
#                 features[col] = features[col].clip(lower=0)
#                 features[col] = np.log1p(features[col])
#     if future_year is not None:
#         features['Year'] = features.index.to_series().str.extract(r'_(\d{4})')[0].astype(int)
#         train_features = features[features['Year'] < future_year]
#     else:
#         train_features = features.copy()
#     means = train_features[feature_cols].mean()
#     stds = train_features[feature_cols].std()
#     stds[stds < 1e-8] = 1e-8
#     features[feature_cols] = (features[feature_cols] - means) / stds
#     if 'Year' in features.columns:
#         features = features.drop(columns=['Year'])
#     return features
# features = normalize_features(features, log_transform_cols=['Traffic'], future_year=2023)

# # Fill temperature with mean
# features['Temperature'] = features['Temperature'].fillna(features['Temperature'].mean())

# # --- Diagnostics ---
# print("\n🔎 Feature Summary Statistics:")
# print(features.describe().transpose())

# print("\n🧪 Checking for any remaining NaN values in features:")
# print(features.isna().sum()[features.isna().sum() > 0])

# print("\n🧪 Checking for any duplicated feature indices (RoadID_Year):")
# print(features.index.duplicated().sum(), "duplicates found")

# print("\n📏 Feature Standard Deviations (should not be near zero):")
# print(features.std().sort_values())

# print("\n📊 Feature Distributions Snapshot (first few columns):")
# for col in features.columns[:5]:
#     print(f"\n--- {col} ---")
#     print(f"Min: {features[col].min():.4f}, Max: {features[col].max():.4f}")
#     print(f"Skewness: {features[col].skew():.4f}, Kurtosis: {features[col].kurtosis():.4f}")
#     print(features[col].hist(bins=30).get_figure())  # Only if using Jupyter

# # --- Labels diagnostics ---
# print("\n🧾 Label Summary:")
# print(labels.describe())

# print("\n🔍 Unique Label Values:")
# print(labels.value_counts(dropna=False))

# if labels.shape[1] == 1 and labels.iloc[:, 0].nunique() <= 10:
#     print("\n📊 Label Class Distribution (if classification):")
#     print(labels.iloc[:, 0].value_counts(normalize=True).round(3))
# elif labels.shape[1] == 1:
#     print("\n📊 Label Histogram (if regression):")
#     labels.iloc[:, 0].hist(bins=30)

# # --- Cross-check IDs ---
# print("\n🔗 Sanity Check: Number of nodes in features:", features.shape[0])
# print("🔗 Number of nodes with labels:", labels.shape[0])
# print("🔗 Number of edges:", edges.shape[0])
# print("🔗 Intersection of features and labels:", len(set(features.index) & set(labels.index)))

# # --- Print mismatched feature and label IDs ---
# feature_ids = set(features.index)
# label_ids = set(labels.index)

# only_in_features = feature_ids - label_ids
# only_in_labels = label_ids - feature_ids

# if only_in_features:
#     print(f"\n❌ {len(only_in_features)} IDs only in features:")
#     for id in sorted(only_in_features):
#         print("  Feature only:", id)

# if only_in_labels:
#     print(f"\n❌ {len(only_in_labels)} IDs only in labels:")
#     for id in sorted(only_in_labels):
#         print("  Label only:", id)

# if not only_in_features and not only_in_labels:
#     print("\n✅ All features and labels align perfectly.")
