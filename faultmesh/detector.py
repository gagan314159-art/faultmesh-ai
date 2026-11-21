"""Anomaly detector module for FaultMesh AI.

Responsibility:
    Isolation Forest anomaly detection on engineered multi-sensor features,
    combining isolation path depth with standardized Z-score deviations into calibrated
    [0, 1] anomaly scores, and evaluating consecutive-reading alarm onset filters.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from faultmesh.processing import add_features


def first_onset(
    scores: pd.DataFrame,
    threshold: float = 0.6,
    min_consecutive: int = 3,
) -> Dict[str, Optional[int]]:
    """Find the first time step where each sensor's score stays above threshold for consecutive readings.

    Args:
        scores: DataFrame containing anomaly scores with columns like 'S1', 'S2', 'S3', 'S4', 'S5',
                and optional 'time' column for time-step indexing.
        threshold: Anomaly score cutoff required to flag abnormal behavior (default: 0.6).
        min_consecutive: Number of consecutive samples exceeding threshold to confirm onset (default: 3).

    Returns:
        Dict mapping each sensor ID to its first detection onset time step (or None if no alarm occurred).
    """
    time_cols = {"time", "timestep"}
    sensor_cols = [c for c in scores.columns if c not in time_cols and not c.startswith("_")]
    onsets: Dict[str, Optional[int]] = {}

    has_time = "time" in scores.columns or "timestep" in scores.columns
    time_col = "time" if "time" in scores.columns else ("timestep" if "timestep" in scores.columns else None)

    for sensor in sensor_cols:
        series = scores[sensor].values
        n_samples = len(series)
        onset_step: Optional[int] = None
        consecutive = 0

        for idx in range(n_samples):
            if series[idx] >= threshold:
                consecutive += 1
                if consecutive >= min_consecutive:
                    # Onset represents the start of the sustained abnormal run
                    start_idx = idx - min_consecutive + 1
                    onset_step = int(scores[time_col].iloc[start_idx]) if has_time else start_idx
                    break
            else:
                consecutive = 0

        onsets[sensor] = onset_step

    return onsets


class AnomalyDetector:
    """Manages sensor-level Isolation Forest detectors and multi-criteria anomaly scoring.

    Isolation Forest Architecture:
        - Trained individually per sensor on clean baseline operating samples (t=0..99).
        - Feature Space per Sensor: [roll_mean, roll_std, diff, zscore].
        - Parameters:
            * n_estimators=100: Robust ensemble size providing stable average path lengths.
            * max_samples='auto': Standard subsampling avoiding memory overhead on time series.
            * contamination=0.01: Very low outlier expectation on clean normal baseline window.
            * random_state=42: Deterministic tree partitioning for reproducible evaluations.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        consecutive_required: int = 3,
        random_state: int = 42,
        n_estimators: int = 100,
        contamination: float = 0.01,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            threshold: Anomaly score threshold in [0, 1] for alarm declaration (default: 0.6).
            consecutive_required: Consecutive abnormal samples required to confirm onset (default: 3).
            random_state: Seed for random partitioning trees (default: 42).
            n_estimators: Number of isolation trees per sensor model (default: 100).
            contamination: Expected fraction of outliers in training set (default: 0.01).
        """
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.contamination = contamination

        self.models: Dict[str, IsolationForest] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.sensors: List[str] = ["S1", "S2", "S3", "S4", "S5"]

    def fit(self, normal_df: pd.DataFrame) -> None:
        """Train one Isolation Forest model per sensor on normal baseline features (first 100 samples).

        Args:
            normal_df: DataFrame containing sensor readings or pre-computed features.
                       Uses the first 100 rows for model fitting.
        """
        # Ensure engineered features are present
        if not any(f"{s}_roll_mean" in normal_df.columns for s in self.sensors):
            feats_df = add_features(normal_df)
        else:
            feats_df = normal_df

        # Restrict training strictly to first 100 normal baseline samples
        baseline_train = feats_df.iloc[:100]

        for sensor in self.sensors:
            cols = [
                f"{sensor}_roll_mean",
                f"{sensor}_roll_std",
                f"{sensor}_diff",
                f"{sensor}_zscore",
            ]
            self.feature_names[sensor] = cols

            model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples="auto",
                contamination=self.contamination,
                random_state=self.random_state,
            )
            model.fit(baseline_train[cols])
            self.models[sensor] = model

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute calibrated anomaly scores between 0 and 1 for every sensor and time step.

        COMBINATION AND NORMALIZATION FORMULA:
        --------------------------------------
        1. Isolation Forest Output:
           raw_decision = model.decision_function(X)
           - For normal inlier data, decision_function is positive (approx +0.05 to +0.25).
           - For anomalies, decision_function is negative (approx -0.10 to -0.35).
           - Decision threshold is 0.0.
           We map this into [0, 1] via:
               iforest_score = np.clip(0.5 - 2.0 * raw_decision, 0.0, 1.0)
           * When raw_decision == 0.0 -> iforest_score = 0.50 (neutral/boundary)
           * When raw_decision > 0.0  -> iforest_score < 0.50 (normal inlier, ~0.1 to 0.3)
           * When raw_decision < 0.0  -> iforest_score > 0.50 (abnormal outlier, ~0.7 to 1.0)

        2. Standardized Z-Score Component:
           Z = (x_t - baseline_mean) / baseline_std
           In normal Gaussian data, |Z| < 3.0 for ~99.7% of samples.
           We normalize |Z| to [0, 1] via:
               z_norm = np.clip(|Z| / 4.0, 0.0, 1.0)
           * |Z| = 0.0 -> z_norm = 0.00
           * |Z| = 1.0 -> z_norm = 0.25 (typical normal noise)
           * |Z| = 3.0 -> z_norm = 0.75 (borderline anomaly)
           * |Z| >= 4.0 -> z_norm = 1.00 (severe physical fault)

        3. Composite Anomaly Score:
           composite_score = 0.50 * iforest_score + 0.50 * z_norm
           * Clean normal data: composite_score ~ 0.15 - 0.25 (well below 0.6 threshold).
           * Abnormal faulty data: composite_score >= 0.60 - 1.00.

        Args:
            df: Input DataFrame of raw readings or extracted features.

        Returns:
            DataFrame with 'time' column and sensor anomaly scores in [0, 1].
        """
        # Ensure engineered features are available
        if not any(f"{s}_roll_mean" in df.columns for s in self.sensors):
            feats_df = add_features(df)
        else:
            feats_df = df

        out = pd.DataFrame()
        if "time" in df.columns:
            out["time"] = df["time"].values
        elif "timestep" in df.columns:
            out["time"] = df["timestep"].values

        for sensor in self.sensors:
            if sensor not in self.models:
                raise RuntimeError(f"Detector for sensor '{sensor}' has not been fitted. Call fit() first.")

            cols = self.feature_names[sensor]
            raw_decision = self.models[sensor].decision_function(feats_df[cols])

            # 1. Isolation Forest normalized score
            iforest_score = np.clip(0.5 - 2.0 * raw_decision, 0.0, 1.0)

            # 2. Z-score normalized score
            z_raw = feats_df[f"{sensor}_zscore"].values
            z_norm = np.clip(np.abs(z_raw) / 4.0, 0.0, 1.0)

            # 3. Equal-weight combination (both in [0, 1], composite in [0, 1])
            composite = 0.5 * iforest_score + 0.5 * z_norm

            # Safeguard against any unexpected NaN / infinite values
            safe_scores = np.nan_to_num(composite, nan=0.0, posinf=1.0, neginf=0.0)
            out[sensor] = np.clip(safe_scores, 0.0, 1.0)

        return out

    def first_onset(
        self,
        scores: pd.DataFrame,
        threshold: Optional[float] = None,
        min_consecutive: Optional[int] = None,
    ) -> Dict[str, Optional[int]]:
        """Find first detection onset time step for each sensor."""
        th = threshold if threshold is not None else self.threshold
        mc = min_consecutive if min_consecutive is not None else self.consecutive_required
        return first_onset(scores, threshold=th, min_consecutive=mc)

    # -------------------------------------------------------------------------
    # Backward Compatibility Methods for pipeline.py and tests/test_scenarios.py
    # -------------------------------------------------------------------------
    def fit_sensor_model(
        self,
        sensor_id: str,
        baseline_features: pd.DataFrame,
        feature_columns: List[str],
    ) -> None:
        """Compatibility method: fit single sensor Isolation Forest."""
        self.feature_names[sensor_id] = feature_columns
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(baseline_features[feature_columns].iloc[:100])
        self.models[sensor_id] = model

    def fit_all(
        self,
        baseline_df: pd.DataFrame,
        sensor_ids: List[str],
    ) -> None:
        """Compatibility method: fit multiple sensors."""
        self.sensors = list(sensor_ids)
        self.fit(baseline_df)

    def compute_anomaly_scores(
        self,
        df: pd.DataFrame,
        sensor_id: str,
    ) -> np.ndarray:
        """Compatibility method: compute 1D anomaly scores for single sensor."""
        scores_df = self.score(df)
        return scores_df[sensor_id].values

    def evaluate_detections(
        self,
        df: pd.DataFrame,
        sensor_ids: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Optional[int]]]:
        """Compatibility method: compute scores and alarm flags for pipeline."""
        scores_df = self.score(df)
        result = df.copy()
        first_alarm_steps: Dict[str, Optional[int]] = {}

        for sensor in sensor_ids:
            scores = scores_df[sensor].values
            result[f"{sensor}_anomaly_score"] = scores
            alarm_flags = np.zeros(len(df), dtype=bool)

            consecutive = 0
            first_alarm: Optional[int] = None
            for t, val in enumerate(scores):
                if val >= self.threshold:
                    consecutive += 1
                    if consecutive >= self.consecutive_required:
                        alarm_flags[t] = True
                        if first_alarm is None:
                            first_alarm = t - self.consecutive_required + 1
                else:
                    consecutive = 0

            result[f"{sensor}_alarm"] = alarm_flags
            first_alarm_steps[sensor] = first_alarm

        return result, first_alarm_steps
