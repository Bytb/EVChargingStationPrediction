import numpy as np
import pandas as pd

edges = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Raw_Edges_Expanded.csv")  # source and target are RoadID_Year
features = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Raw_Features_Test.csv", index_col=0)
labels = pd.read_csv(r"C:\Users\Caleb\OneDrive - University of South Florida\EV_Research\EV_Research_PythonCode\data\tampa\Tampa_Labels_Test.csv", index_col=0)  # also indexed by RoadID_Year

# --- Load expanded edge list with RoadID_Year format ---
def adjust_edge(entry):
    road, year = entry.replace("Road", "").split("_")
    return f"Road{int(road)-1}_{year}"

edges['source'] = edges['source'].apply(adjust_edge)
edges['target'] = edges['target'].apply(adjust_edge)

# --- Load features and labels (index_col=0 to keep 'RoadID_Year' as index) ---
def adjust_road_id(entry):
    road, year = entry.replace("Road", "").split("_")
    return f"Road{int(road)-1}_{year}"

# Apply to index
features.index = features.index.to_series().apply(adjust_road_id)

def normalize_features(features, log_transform_cols=None, future_year=None):
    """
    Normalizes features with optional log-transform and future-year holdout protection.
    
    Parameters:
    - features: DataFrame indexed by RoadID_Year format (e.g., 'Road123_2019')
    - log_transform_cols: list of columns to apply log1p (e.g., ['Traffic'])
    - future_year: int (e.g., 2025) if you want to exclude future years from training stats

    Returns:
    - normalized DataFrame
    """

    feature_cols = features.columns.tolist()

    # Optional log-transform
    if log_transform_cols:
        for col in log_transform_cols:
            if col in features.columns:
                features[col] = features[col].clip(lower=0)
                features[col] = np.log1p(features[col])

    # Extract year if needed
    if future_year is not None:
        features['Year'] = features.index.to_series().str.extract(r'_(\d{4})')[0].astype(int)
        train_features = features[features['Year'] < future_year]
    else:
        train_features = features.copy()

    # Compute stats
    means = train_features[feature_cols].mean()
    stds = train_features[feature_cols].std()
    stds[stds < 1e-8] = 1e-8  # prevent divide-by-zero

    # Apply normalization
    features[feature_cols] = (features[feature_cols] - means) / stds

    # Drop helper column if added
    if 'Year' in features.columns:
        features = features.drop(columns=['Year'])

    return features

#normalize data
features = normalize_features(
    features,
    log_transform_cols=['Traffic'],
    future_year=None  # or 2025 if you add test data later
)