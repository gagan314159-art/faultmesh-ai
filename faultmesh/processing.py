"""Feature processing module for FaultMesh AI.

Responsibility:
    Calculate rolling statistics and detection features (rolling mean, rolling std,
    first difference, and Z-scores) from raw multi-sensor time-series readings.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def add_features(
    df: pd.DataFrame,
    window: int = 10,
    baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Calculate rolling statistics and standardized features for every sensor.

    For each sensor in the DataFrame (e.g. S1, S2, S3, S4, S5), computes:
        1. Rolling mean: Local moving average across `window` steps (min_periods=1).
        2. Rolling standard deviation: Local volatility across `window` steps.
        3. First difference: Consecutive sample delta (x_t - x_{t-1}), filled with 0.0 at t=0.
        4. Z-score: Standardized deviation relative to calibrated baseline statistics
                    (or the first 100 normal samples when baseline_stats is None):
                    Z = (x_t - baseline_mean) / baseline_std.
                    Safe against division-by-zero (clips baseline_std to >= 1e-6).

    Args:
        df: Input DataFrame containing 'time' and sensor reading columns.
        window: Size of the rolling calculation window (default: 10).
        baseline_stats: Optional dictionary mapping sensor ID to {'mean': float, 'std': float}.
                        When provided, guarantees that live streaming telemetry is evaluated
                        strictly against the verified normal baseline without contamination.

    Returns:
        New pandas DataFrame containing original time and sensor readings, augmented with
        consistent '{sensor}_roll_mean', '{sensor}_roll_std', '{sensor}_diff', and
        '{sensor}_zscore' columns.
    """
    res = df.copy()

    # Identify sensor columns (exclude time / metadata columns)
    time_cols = {"time", "timestep"}
    sensor_cols = [c for c in df.columns if c not in time_cols and not c.startswith("_")]

    for sensor in sensor_cols:
        # 1. Rolling mean (min_periods=1 ensures no leading NaNs)
        res[f"{sensor}_roll_mean"] = res[sensor].rolling(window=window, min_periods=1).mean()

        # 2. Rolling std (min_periods=1, fill initial single-element NaN with 0.0)
        roll_std = res[sensor].rolling(window=window, min_periods=1).std().fillna(0.0)
        res[f"{sensor}_roll_std"] = roll_std

        # 3. First difference between consecutive readings
        res[f"{sensor}_diff"] = res[sensor].diff().fillna(0.0)

        # 4. Z-score: use calibrated baseline stats if provided, fallback to first 100 rows
        if baseline_stats and sensor in baseline_stats:
            base_mean = float(baseline_stats[sensor]["mean"])
            base_std = float(baseline_stats[sensor]["std"])
        else:
            baseline_slice = res[sensor].iloc[:100]
            base_mean = float(baseline_slice.mean()) if len(baseline_slice) > 0 else float(res[sensor].mean())
            base_std = float(baseline_slice.std()) if len(baseline_slice) > 1 else float(res[sensor].std())

        # Safe division guard against zero or NaN standard deviation
        if np.isnan(base_std) or base_std < 1e-6:
            base_std = 1.0

        z_scores = (res[sensor] - base_mean) / base_std
        # Handle any numerical extremes or NaNs safely
        res[f"{sensor}_zscore"] = z_scores.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return res


class FeatureProcessor:
    """Class wrapper providing baseline calibration and feature extraction."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialize feature processor with rolling window parameters."""
        self.window_size = window_size
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit_baseline(self, normal_df: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> None:
        """Calculate mean and standard deviation from normal baseline samples."""
        cols = sensor_cols if sensor_cols is not None else [
            c for c in normal_df.columns if c not in ("time", "timestep") and not c.startswith("_")
        ]
        for col in cols:
            if col in normal_df.columns:
                std_val = float(normal_df[col].std())
                self.baseline_stats[col] = {
                    "mean": float(normal_df[col].mean()),
                    "std": std_val if (not np.isnan(std_val) and std_val > 1e-6) else 1.0,
                }

    def compute_rolling_statistics(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Compute rolling mean and rolling std."""
        result = df.copy()
        for col in sensor_cols:
            result[f"{col}_roll_mean"] = result[col].rolling(window=self.window_size, min_periods=1).mean()
            result[f"{col}_roll_std"] = result[col].rolling(window=self.window_size, min_periods=1).std().fillna(0.0)
        return result

    def compute_first_differences(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Compute first-order differences."""
        result = df.copy()
        for col in sensor_cols:
            result[f"{col}_diff"] = result[col].diff().fillna(0.0)
        return result

    def compute_z_scores(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Compute Z-scores using pre-fitted baseline stats."""
        result = df.copy()
        for col in sensor_cols:
            stats = self.baseline_stats.get(col, {"mean": 0.0, "std": 1.0})
            mean = stats["mean"]
            std = stats["std"] if stats["std"] > 1e-6 else 1.0
            result[f"{col}_zscore"] = (result[col] - mean) / std
        return result

    def extract_features(self, df: pd.DataFrame, sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract complete set of detection features."""
        return add_features(df, window=self.window_size, baseline_stats=self.baseline_stats or None)
