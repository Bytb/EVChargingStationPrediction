# transform_features_nonrnn.py
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# -------- Helper: invert label transform back to raw scale --------
def inverse_transform_preds(preds_np, y_meta):
    if y_meta is None:
        return preds_np

    if isinstance(y_meta, dict) and y_meta.get("method") == "zscore":
        return preds_np * y_meta["std"] + y_meta["mean"]

    if isinstance(y_meta, dict) and y_meta.get("method") == "asinh":
        c = y_meta.get("scale", 1.0)
        if y_meta.get("standardize_after"):
            m, s = y_meta["mean"], y_meta["std"]
            return np.sinh(preds_np * s + m) * c
        else:
            return np.sinh(preds_np) * c

    if hasattr(y_meta, "inverse_transform"):  # yeo-johnson / quantile
        return y_meta.inverse_transform(preds_np.reshape(-1,1)).ravel()

    return preds_np



# --- transforms_labels.py (replace your transform_labels with this) ---
def transform_labels(
    labels_df,
    train_mask,
    method: str = "zscore",
    device: str = "cpu",
    standardize_after: bool = False,
    asinh_scale="iqr",
    **kwargs
):
    """Fit on TRAIN rows only; return transformed y tensor + meta for inverse."""
    y = labels_df.select_dtypes(include=[np.number]).values.squeeze()
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    train_mask_np = train_mask.detach().cpu().numpy().astype(bool)
    y_train = y[train_mask_np]

    # ---- Print BEFORE stats ----
    # print("\n📊 Label Statistics BEFORE Transformation:")
    # before_df = pd.DataFrame({
    #     'min': np.min(y, axis=0),
    #     'max': np.max(y, axis=0),
    #     'mean': np.mean(y, axis=0),
    #     'std': np.std(y, axis=0),
    #     'skew': pd.Series(y.flatten()).skew()
    # })
    # print(before_df.round(4))

    transformer = None

    if method == "zscore":
        mean = y_train.mean(axis=0)
        std  = y_train.std(axis=0)
        std[std < 1e-8] = 1e-8
        y_tx = (y - mean) / std
        transformer = {"method":"zscore", "mean":mean, "std":std}

    elif method == "asinh":
        # determine scale c for pre-scaling (train-only)
        if isinstance(asinh_scale, (int, float)):
            c = float(asinh_scale) if asinh_scale > 0 else 1.0
        elif asinh_scale == "iqr":
            q75, q25 = np.percentile(y_train, [75, 25], axis=0)
            c = np.maximum(q75 - q25, 1e-8)
        elif asinh_scale == "std":
            c = np.maximum(y_train.std(axis=0), 1e-8)
        else:
            c = 1.0

        z = np.arcsinh(y / c)  # pre-scale then asinh

        if standardize_after:
            m = z[train_mask_np].mean(axis=0)
            s = z[train_mask_np].std(axis=0)
            s[s < 1e-8] = 1e-8
            y_tx = (z - m) / s
            transformer = {
                "method":"asinh",
                "standardize_after": True,
                "mean": m, "std": s, "scale": c
            }
        else:
            y_tx = z
            transformer = {
                "method":"asinh",
                "standardize_after": False,
                "scale": c
            }

    elif method == "yeo-johnson":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True, **kwargs)
        transformer.fit(y_train)
        y_tx = transformer.transform(y)

    elif method == "quantile":
        transformer = QuantileTransformer(output_distribution="normal", **kwargs)
        transformer.fit(y_train)
        y_tx = transformer.transform(y)

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- Print AFTER stats ----
    # print("\n📊 Label Statistics AFTER Transformation:")
    # after_df = pd.DataFrame({
    #     'min': np.min(y_tx, axis=0),
    #     'max': np.max(y_tx, axis=0),
    #     'mean': np.mean(y_tx, axis=0),
    #     'std': np.std(y_tx, axis=0),
    #     'skew': pd.Series(y_tx.flatten()).skew()
    # })
    # print(after_df.round(4))

    y_tensor = torch.tensor(y_tx.squeeze(), dtype=torch.float32).to(device)
    return y_tensor, transformer


# === transforms_features_with_mask.py ===
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

def _to_bool_mask(mask: Union[np.ndarray, pd.Series, "torch.Tensor"], n: int) -> np.ndarray:
    """Coerce an incoming mask to a 1D numpy bool array of length n."""
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(mask, pd.Series):
        mask = mask.values
    mask = np.asarray(mask).astype(bool).reshape(-1)
    if mask.shape[0] != n:
        raise ValueError(f"train_mask length {mask.shape[0]} != data length {n}")
    return mask

def _stats_before_after_with_mask(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    train_mask: Union[np.ndarray, pd.Series, "torch.Tensor"],
    neg_clipped_counts: Dict[str, int],
    columns: List[str]
) -> pd.DataFrame:
    """
    Build a console-ready table with before/after stats.
    - Mean/Std are computed on TRAIN rows only (mask=True).
    - Min/Max/Skew are computed on ALL rows.
    """
    is_train = _to_bool_mask(train_mask, len(df_before))

    def stat_block(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        mean_train = df.loc[is_train, columns].mean()
        std_train  = df.loc[is_train, columns].std()
        min_all    = df[columns].min()
        max_all    = df[columns].max()
        skew_all   = df[columns].skew()
        blk = pd.DataFrame({
            f"Mean_{tag}": mean_train,
            f"Std_{tag}":  std_train,
            f"Min_{tag}":  min_all,
            f"Max_{tag}":  max_all,
            f"Skew_{tag}": skew_all,
        })
        return blk

    before_blk = stat_block(df_before, "B")
    after_blk  = stat_block(df_after,  "A")
    out = before_blk.join(after_blk)

    # Negatives clipped report (0 for non-log columns)
    out["NegClip"] = [neg_clipped_counts.get(c, 0) for c in out.index]

    # Pretty rounding for console readability
    out = out[[
        "Mean_B","Std_B","Min_B","Max_B","Skew_B",
        "Mean_A","Std_A","Min_A","Max_A","Skew_A",
        "NegClip"
    ]].round({
        "Mean_B":3,"Std_B":3,"Min_B":3,"Max_B":3,"Skew_B":2,
        "Mean_A":3,"Std_A":3,"Min_A":3,"Max_A":3,"Skew_A":2
    })
    return out

def transform_features(
    features: pd.DataFrame,
    train_mask: Union[np.ndarray, pd.Series, "torch.Tensor"],
    log_cols: Optional[List[str]] = None,
    no_scale_cols: Optional[List[str]] = None,
    temperature_fill: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply:
      - (Optional) train-mean fill for Temperature NaNs (leak-free),
      - log1p to LOG_COLS (with negative clipping & warnings),
      - z-score using TRAIN-ONLY stats for all columns except NO_SCALE_COLS.

    Parameters
    ----------
    features : pd.DataFrame
        Rows must align with the provided train_mask. All columns are treated as features.
    train_mask : array-like of bool
        Boolean mask (len == len(features)) indicating which rows belong to TRAIN.
    log_cols : list[str] or None
        Columns to force log1p. (We clip negatives to 0 and report counts.)
    no_scale_cols : list[str] or None
        Columns to leave unscaled (no z-score). They also skip log unless included in log_cols.
    temperature_fill : bool
        If True and 'Temperature' exists, fill NaNs with TRAIN-ONLY mean.
    verbose : bool
        Print a summary table and warnings.

    Returns
    -------
    df_scaled : pd.DataFrame
        Transformed features (same shape/index/columns as input).
    summary : pd.DataFrame
        Before/after stats table printed to console as well.
    """
    log_cols = list(log_cols or [])
    no_scale_cols = list(no_scale_cols or [])

    n = len(features)
    is_train = _to_bool_mask(train_mask, n)

    # Work on copies so we can compute before/after cleanly
    df = features.copy()
    df_before = df.copy()

    # All columns are considered feature columns here
    feature_cols = list(df.columns)

    # 1) Temperature fill with TRAIN mean (leak-free)
    if temperature_fill and "Temperature" in df.columns:
        train_temp_mean = df.loc[is_train, "Temperature"].mean()
        df["Temperature"] = df["Temperature"].fillna(train_temp_mean)

    # 2) Log transforms for selected columns (with negative clipping report)
    neg_clipped_counts: Dict[str, int] = {}
    existing_log_cols = [c for c in log_cols if c in feature_cols]
    missing_log_cols  = [c for c in log_cols if c not in feature_cols]

    for col in existing_log_cols:
        neg_mask = df[col] < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0 and verbose:
            print(f"⚠ Feature '{col}' has {neg_count} negative values. Clipping to 0 before log1p.")
        if neg_count > 0:
            neg_clipped_counts[col] = neg_count
            df.loc[neg_mask, col] = 0.0
        # safe log1p
        df[col] = np.log1p(df[col])

    if missing_log_cols and verbose:
        print(f"ℹ Skipped log1p for missing columns: {missing_log_cols}")

    # 3) Train-only z-score scaling for scalable columns
    scalable_cols = [c for c in feature_cols if c not in no_scale_cols]

    # Compute means/stds on TRAIN subset AFTER log where applicable
    means = df.loc[is_train, scalable_cols].mean()
    stds  = df.loc[is_train, scalable_cols].std()
    stds_replaced = stds.copy()
    stds_replaced[stds_replaced < 1e-8] = 1e-8

    # Apply scaling to ALL rows
    df_scaled = df.copy()
    df_scaled[scalable_cols] = (df_scaled[scalable_cols] - means) / stds_replaced

    # 4) Build and (optionally) print summary table
    summary = _stats_before_after_with_mask(
        df_before=df_before,
        df_after=df_scaled,
        train_mask=is_train,
        neg_clipped_counts=neg_clipped_counts,
        columns=feature_cols
    )

    if verbose:
        print("\n=== Transform Summary (Features) ===")
        print(f"Log1p columns (applied): {existing_log_cols}")
        if missing_log_cols:
            print(f"Log1p columns (not found): {missing_log_cols}")
        print(f"No-scale columns: {no_scale_cols}")
        print("\nFeature stats (Before/After; train means/stds, all-years min/max/skew):")
        with pd.option_context('display.max_rows', None, 'display.width', 140):
            print(summary)

        clipped_any = {k: v for k, v in neg_clipped_counts.items() if v > 0}
        if clipped_any:
            print("\nNegatives clipped before log1p:")
            for k, v in clipped_any.items():
                print(f"  - {k}: {v}")

    return df_scaled, summary