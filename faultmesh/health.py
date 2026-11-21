"""Sensor Health Monitoring Engine for FaultMesh AI 2.0.

Responsibility:
    Evaluates physical sensor operational condition and fleet-wide network health
    using multi-dimensional telemetry analysis: anomaly frequency, baseline drift,
    variance stability, sustained fault duration, and recent trend trajectories.
    Categorizes health into 4 transparent states:
    - Healthy (85–100)
    - Warning (65–84)
    - Critical (0–64)
    - Insufficient Data
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from faultmesh.simulator import DEFAULT_BASELINES


SENSOR_METADATA: Dict[str, Dict[str, str]] = {
    "S1": {"name": "Bearing Temperature", "unit": "°C", "system": "Bearing Housing"},
    "S2": {"name": "Shaft Vibration", "unit": "mm/s", "system": "Drive Shaft"},
    "S3": {"name": "Cooling Pressure", "unit": "bar", "system": "Cooling Manifold"},
    "S4": {"name": "Coolant Flow", "unit": "L/min", "system": "Coolant Delivery"},
    "S5": {"name": "Motor Current", "unit": "A", "system": "Drive Motor Phase"},
}

HEALTH_DISCLAIMER: str = (
    "NOTICE: Sensor health scores reflect simulated operational telemetry and anomaly patterns. "
    "They represent diagnostic health indicators and do NOT constitute physical run-to-failure "
    "reliability probabilities."
)


class SensorHealthRecord:
    """Individual sensor health evaluation report."""

    def __init__(
        self,
        sensor_id: Optional[str] = None,
        *args: Any,
        health_score: Union[float, int, str] = 100.0,
        health_status: Optional[str] = None,
        anomaly_frequency: float = 0.0,
        average_anomaly_score: float = 0.0,
        maximum_anomaly_score: float = 0.0,
        detection_onset: Optional[int] = None,
        recent_trend: str = "Stable",
        health_explanation: str = "",
        recommended_monitoring_priority: str = "Normal",
        current_value: Optional[float] = None,
        baseline_value: float = 0.0,
        unit: str = "",
        sensor_name: str = "",
        degradation_factors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize sensor health record with flexible arguments supporting legacy positional calls."""
        # Support legacy positional signature: (sensor, sensor_name, health_score, status, current_value, baseline, unit, ...)
        if args and len(args) >= 1:
            if isinstance(args[0], str) and len(args) >= 2 and isinstance(args[1], (int, float)):
                sensor_name = args[0]
                health_score = args[1]
                if len(args) >= 3:
                    health_status = args[2]
                if len(args) >= 4:
                    current_value = args[3]
                if len(args) >= 5:
                    baseline_value = args[4]
                if len(args) >= 6:
                    unit = args[5]
                if len(args) >= 7:
                    anomaly_frequency = args[6]
                if len(args) >= 10:
                    degradation_factors = args[9]
            elif isinstance(args[0], (int, float)):
                health_score = args[0]
                if len(args) >= 2:
                    health_status = args[1]

        resolved_sensor = sensor_id or kwargs.get("sensor", "")
        self.sensor_id: str = str(resolved_sensor)

        # Parse health_score safely
        try:
            self.health_score: float = float(health_score)
        except (ValueError, TypeError):
            self.health_score = 0.0

        raw_status = health_status or kwargs.get("status", "Healthy")
        raw_status_str = str(raw_status).strip()
        if raw_status_str.upper() == "HEALTHY":
            self.health_status: str = "Healthy"
        elif raw_status_str.upper() == "WARNING":
            self.health_status = "Warning"
        elif raw_status_str.upper() == "CRITICAL":
            self.health_status = "Critical"
        elif "INSUFFICIENT" in raw_status_str.upper():
            self.health_status = "Insufficient Data"
        else:
            self.health_status = raw_status_str.capitalize()

        self.anomaly_frequency: float = float(anomaly_frequency)
        self.average_anomaly_score: float = float(average_anomaly_score)
        self.maximum_anomaly_score: float = float(maximum_anomaly_score)
        self.detection_onset: Optional[int] = detection_onset
        self.recent_trend: str = str(recent_trend)
        self.health_explanation: str = str(health_explanation)
        self.recommended_monitoring_priority: str = str(recommended_monitoring_priority)
        self.current_value: Optional[float] = float(current_value) if current_value is not None else None
        self.baseline_value: float = float(baseline_value)
        self.unit: str = str(unit)
        self.sensor_name: str = str(sensor_name or kwargs.get("sensor_name", self.sensor_id))
        self.degradation_factors: List[str] = list(degradation_factors) if degradation_factors is not None else []

    def __repr__(self) -> str:
        return (
            f"SensorHealthRecord(sensor_id={self.sensor_id!r}, health_score={self.health_score}, "
            f"health_status={self.health_status!r}, priority={self.recommended_monitoring_priority!r})"
        )

    # Aliases for backward compatibility
    @property
    def sensor(self) -> str:
        return self.sensor_id

    @property
    def status(self) -> str:
        return self.health_status

    @property
    def max_zscore(self) -> float:
        for f in self.degradation_factors:
            if "σ" in f:
                try:
                    part = f.split("σ")[0].split()[-1]
                    part = part.replace("peak", "").strip()
                    return float(part)
                except Exception:
                    pass
        return 0.0

    @property
    def longest_alarm_run(self) -> int:
        for f in self.degradation_factors:
            if "consecutive" in f or "samples" in f:
                try:
                    for token in f.split():
                        if token.isdigit():
                            return int(token)
                except Exception:
                    pass
        return 0


# Alias for backward compatibility
SensorHealthReport = SensorHealthRecord


@dataclass
class NetworkHealthSummary:
    """Network-level health aggregation summary."""
    overall_network_health_score: float     # 0.0 to 100.0 (mean of valid sensor scores)
    healthy_count: int
    warning_count: int
    critical_count: int
    insufficient_data_count: int
    most_vulnerable_sensor: Optional[str]
    highest_priority_sensor: Optional[str]
    sensor_reports: Dict[str, SensorHealthRecord] = field(default_factory=dict)
    disclaimer: str = HEALTH_DISCLAIMER


class SensorHealthMonitor:
    """Evaluates operational health of sensor nodes and the overall mesh network."""

    def __init__(
        self,
        baselines: Optional[Dict[str, float]] = None,
        noise_ratio: float = 0.015,
        anomaly_threshold: float = 0.60,
        healthy_threshold: float = 85.0,
        warning_threshold: float = 65.0,
        min_samples_required: int = 10,
    ) -> None:
        """Initialize health monitor with configurable scoring parameters.

        Args:
            baselines: Dict mapping sensor ID to nominal baseline value.
            noise_ratio: Expected nominal noise ratio (default: 0.015 = 1.5%).
            anomaly_threshold: Anomaly score cutoff for declaring abnormal sample (default: 0.60).
            healthy_threshold: Minimum score for Healthy status (default: 85.0).
            warning_threshold: Minimum score for Warning status (default: 65.0).
            min_samples_required: Minimum samples required for health assessment (default: 10).
        """
        self.baselines = baselines.copy() if baselines is not None else DEFAULT_BASELINES.copy()
        self.noise_ratio = noise_ratio
        self.anomaly_threshold = anomaly_threshold
        self.healthy_threshold = healthy_threshold
        self.warning_threshold = warning_threshold
        self.min_samples_required = min_samples_required

    def assess_sensor(
        self,
        sensor: str,
        readings: Optional[Union[pd.Series, np.ndarray, List[float]]],
        scores: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        detection_onset: Optional[int] = None,
    ) -> SensorHealthRecord:
        """Evaluate operational health for an individual sensor channel.

        Args:
            sensor: Sensor identifier (e.g., 'S1').
            readings: Series/array of raw sensor telemetry readings over time.
            scores: Optional Series/array of calibrated anomaly scores (0 to 1).
            detection_onset: Optional first confirmed detection onset timestep.

        Returns:
            SensorHealthRecord with health score, status, and diagnostic metrics.
        """
        meta = SENSOR_METADATA.get(sensor, {"name": sensor, "unit": "", "system": "Subsystem"})
        sensor_name = meta["name"]
        unit = meta["unit"]
        baseline = self.baselines.get(sensor, 1.0)
        expected_sigma = abs(baseline * self.noise_ratio) if baseline != 0 else 0.01

        # Handle None or empty readings
        if readings is None:
            return self._insufficient_data_record(sensor, sensor_name, unit, baseline, "Telemetry data stream is missing (None).")

        # Convert to clean numeric array
        if isinstance(readings, pd.Series):
            valid_readings = readings.dropna().values
        elif isinstance(readings, np.ndarray):
            valid_readings = readings[~np.isnan(readings)]
        elif isinstance(readings, list):
            valid_readings = np.array([x for x in readings if x is not None and not np.isnan(x)], dtype=float)
        else:
            return self._insufficient_data_record(sensor, sensor_name, unit, baseline, "Invalid telemetry data type.")

        if len(valid_readings) < self.min_samples_required:
            return self._insufficient_data_record(
                sensor, sensor_name, unit, baseline,
                f"Fewer than {self.min_samples_required} valid telemetry samples available ({len(valid_readings)} found)."
            )

        vals = valid_readings
        curr_val = float(vals[-1])
        degradation_factors: List[str] = []

        # 1. Anomaly Scores & Frequency Analysis
        score_vals = np.array([], dtype=float)
        if scores is not None:
            if isinstance(scores, pd.Series):
                score_vals = scores.fillna(0.0).values
            elif isinstance(scores, np.ndarray):
                score_vals = np.nan_to_num(scores, nan=0.0)
            elif isinstance(scores, list):
                score_vals = np.array([x if x is not None and not np.isnan(x) else 0.0 for x in scores], dtype=float)

        if len(score_vals) > 0:
            avg_score = float(np.mean(score_vals))
            max_score = float(np.max(score_vals))
            abnormal_mask = score_vals >= self.anomaly_threshold
            anomaly_freq = float(np.mean(abnormal_mask))

            # Longest consecutive abnormal run
            max_run = 0
            cur_run = 0
            for is_abn in abnormal_mask:
                if is_abn:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0
            longest_run = max_run
        else:
            avg_score = 0.0
            max_score = 0.0
            anomaly_freq = 0.0
            longest_run = 0

        # 2. Baseline Deviation Analysis (Z-score)
        deviations = np.abs(vals - baseline)
        max_dev = float(np.max(deviations))
        max_zscore = float(max_dev / expected_sigma) if expected_sigma > 0 else 0.0

        recent_n = max(5, int(len(vals) * 0.20))
        recent_window = vals[-recent_n:]
        recent_mean = float(np.mean(recent_window))
        recent_dev_sigma = float(abs(recent_mean - baseline) / expected_sigma) if expected_sigma > 0 else 0.0

        # 3. Variance & Freeze Analysis
        check_win = vals[-min(30, len(vals)):]
        win_std = float(np.std(check_win))
        is_stuck = bool(win_std < (expected_sigma * 0.05) and len(check_win) >= 15)

        # 4. Recent Trend Trajectory
        if len(vals) >= 20:
            first_half = vals[:len(vals) // 2]
            second_half = vals[len(vals) // 2:]
            first_dev = np.mean(np.abs(first_half - baseline))
            second_dev = np.mean(np.abs(second_half - baseline))
            dev_change = second_dev - first_dev

            if is_stuck:
                recent_trend = "Signal Freeze"
            elif max_score >= 0.80 or recent_dev_sigma >= 8.0:
                recent_trend = "Critical Divergence"
            elif dev_change >= (expected_sigma * 2.0):
                recent_trend = "Degrading"
            elif dev_change <= -(expected_sigma * 2.0):
                recent_trend = "Improving"
            else:
                recent_trend = "Stable"
        else:
            recent_trend = "Stable"

        # 5. Penalty-Based Quantitative Health Scoring (100 Base)
        # A. Anomaly Frequency Penalty (up to 35 pts)
        freq_penalty = min(35.0, anomaly_freq * 70.0)
        if anomaly_freq >= 0.35:
            degradation_factors.append(f"Persistent anomaly activity ({anomaly_freq * 100:.1f}% of timesteps exceeding threshold {self.anomaly_threshold})")
        elif anomaly_freq >= 0.10:
            degradation_factors.append(f"Intermittent anomaly activity ({anomaly_freq * 100:.1f}% of operating window)")

        # B. Baseline Drift Penalty (up to 35 pts)
        dev_penalty = 0.0
        if max_zscore >= 10.0 or recent_dev_sigma >= 6.0:
            dev_penalty = 35.0
            degradation_factors.append(f"Severe baseline offset (peak {max_zscore:.1f}σ, recent mean {recent_dev_sigma:.1f}σ from nominal {baseline} {unit})")
        elif max_zscore >= 5.0 or recent_dev_sigma >= 3.0:
            dev_penalty = 22.0
            degradation_factors.append(f"Elevated baseline deviation (peak {max_zscore:.1f}σ from nominal {baseline} {unit})")
        elif max_zscore >= 3.0:
            dev_penalty = 10.0
            degradation_factors.append(f"Mild transient deviation (peak {max_zscore:.1f}σ)")

        # C. Anomaly Intensity Penalty (up to 15 pts)
        intensity_penalty = min(15.0, avg_score * 20.0)

        # D. Signal Freeze Penalty (up to 25 pts)
        freeze_penalty = 0.0
        if is_stuck:
            freeze_penalty = 25.0
            degradation_factors.append(f"Signal freeze detected: near-zero variance (std={win_std:.5f}) indicates transducer lockup.")

        # E. Longest Run Penalty (up to 10 pts)
        run_penalty = 0.0
        if longest_run >= 40:
            run_penalty = 10.0
            degradation_factors.append(f"Prolonged continuous abnormal sequence ({longest_run} consecutive samples)")
        elif longest_run >= 15:
            run_penalty = 5.0

        total_penalties = freq_penalty + dev_penalty + intensity_penalty + freeze_penalty + run_penalty
        health_score = max(0.0, min(100.0, 100.0 - total_penalties))
        health_score = round(health_score, 1)

        # 6. Health Classification
        if health_score >= self.healthy_threshold:
            health_status = "Healthy"
            monitoring_priority = "Normal"
            if not degradation_factors:
                degradation_factors.append(f"Nominal operating telemetry within expected Gaussian noise bounds (peak Z-score {max_zscore:.1f}σ).")
        elif health_score >= self.warning_threshold:
            health_status = "Warning"
            monitoring_priority = "Elevated" if health_score >= 75.0 else "High"
        else:
            health_status = "Critical"
            monitoring_priority = "Immediate Inspection"

        # 7. Explainable Health Narrative
        if health_status == "Healthy":
            health_explanation = (
                f"Sensor {sensor} ({sensor_name}) is Healthy (Health Score: {health_score:.1f}/100). "
                f"Telemetry remains well within nominal baseline limits ({baseline} {unit}). "
                f"Average anomaly score is {avg_score:.3f} with zero sustained alarm runs."
            )
        elif health_status == "Warning":
            health_explanation = (
                f"Sensor {sensor} ({sensor_name}) is in Warning condition (Health Score: {health_score:.1f}/100). "
                f"Telemetry exhibits moderate deviation from baseline (peak {max_zscore:.1f}σ) "
                f"with an anomaly frequency of {anomaly_freq * 100:.1f}%. Close monitoring recommended."
            )
        else:
            reasons_str = "; ".join(degradation_factors[:2]) if degradation_factors else "Severe anomaly pattern"
            health_explanation = (
                f"Sensor {sensor} ({sensor_name}) is in Critical condition (Health Score: {health_score:.1f}/100). "
                f"Primary degradation drivers: {reasons_str}. Recommended action: {monitoring_priority}."
            )

        return SensorHealthRecord(
            sensor_id=sensor,
            health_score=health_score,
            health_status=health_status,
            anomaly_frequency=round(anomaly_freq, 3),
            average_anomaly_score=round(avg_score, 3),
            maximum_anomaly_score=round(max_score, 3),
            detection_onset=detection_onset,
            recent_trend=recent_trend,
            health_explanation=health_explanation,
            recommended_monitoring_priority=monitoring_priority,
            current_value=round(curr_val, 3),
            baseline_value=baseline,
            unit=unit,
            sensor_name=sensor_name,
            degradation_factors=degradation_factors,
        )

    def _insufficient_data_record(
        self,
        sensor: str,
        sensor_name: str,
        unit: str,
        baseline: float,
        reason: str,
    ) -> SensorHealthRecord:
        """Create a standard Insufficient Data record."""
        return SensorHealthRecord(
            sensor_id=sensor,
            health_score=0.0,
            health_status="Insufficient Data",
            anomaly_frequency=0.0,
            average_anomaly_score=0.0,
            maximum_anomaly_score=0.0,
            detection_onset=None,
            recent_trend="Unknown",
            health_explanation=f"Sensor {sensor} ({sensor_name}): Insufficient data available to evaluate operational health. {reason}",
            recommended_monitoring_priority="Verify Telemetry",
            current_value=None,
            baseline_value=baseline,
            unit=unit,
            sensor_name=sensor_name,
            degradation_factors=[reason],
        )

    def assess_all(
        self,
        readings_df: Optional[pd.DataFrame],
        scores_df: Optional[pd.DataFrame] = None,
        onsets: Optional[Dict[str, Optional[int]]] = None,
    ) -> Dict[str, SensorHealthRecord]:
        """Evaluate operational health for all configured sensors in network.

        Args:
            readings_df: DataFrame of raw sensor telemetry readings over time.
            scores_df: Optional DataFrame of calibrated anomaly scores.
            onsets: Optional Dict mapping sensor ID to first alarm onset.

        Returns:
            Dict mapping sensor ID to SensorHealthRecord.
        """
        reports: Dict[str, SensorHealthRecord] = {}
        sensor_keys = list(self.baselines.keys())

        if readings_df is None or (isinstance(readings_df, pd.DataFrame) and readings_df.empty):
            for s in sensor_keys:
                meta = SENSOR_METADATA.get(s, {"name": s, "unit": ""})
                reports[s] = self._insufficient_data_record(
                    s, meta["name"], meta["unit"], self.baselines.get(s, 0.0),
                    "No readings DataFrame provided."
                )
            return reports

        onsets_clean = onsets or {}

        for sensor in sensor_keys:
            if sensor in readings_df.columns:
                r_series = readings_df[sensor]
                s_series = scores_df[sensor] if scores_df is not None and sensor in scores_df.columns else None
                onset = onsets_clean.get(sensor)
                reports[sensor] = self.assess_sensor(sensor, r_series, s_series, detection_onset=onset)
            else:
                meta = SENSOR_METADATA.get(sensor, {"name": sensor, "unit": ""})
                reports[sensor] = self._insufficient_data_record(
                    sensor, meta["name"], meta["unit"], self.baselines.get(sensor, 0.0),
                    f"Sensor channel {sensor} not found in telemetry stream."
                )

        return reports

    def summarize_network_health(
        self,
        reports: Dict[str, SensorHealthRecord],
    ) -> NetworkHealthSummary:
        """Calculate network-level health aggregation metrics across all sensor reports.

        Args:
            reports: Dict mapping sensor ID to SensorHealthRecord.

        Returns:
            NetworkHealthSummary with aggregated metrics, counts, and vulnerable sensors.
        """
        if not reports:
            return NetworkHealthSummary(
                overall_network_health_score=0.0,
                healthy_count=0,
                warning_count=0,
                critical_count=0,
                insufficient_data_count=0,
                most_vulnerable_sensor=None,
                highest_priority_sensor=None,
                sensor_reports={},
                disclaimer=HEALTH_DISCLAIMER,
            )

        valid_reports = [r for r in reports.values() if r.health_status != "Insufficient Data"]

        healthy_cnt = sum(1 for r in reports.values() if r.health_status == "Healthy")
        warning_cnt = sum(1 for r in reports.values() if r.health_status == "Warning")
        critical_cnt = sum(1 for r in reports.values() if r.health_status == "Critical")
        insufficient_cnt = sum(1 for r in reports.values() if r.health_status == "Insufficient Data")

        if valid_reports:
            overall_score = round(float(np.mean([r.health_score for r in valid_reports])), 1)
            # Most vulnerable is the sensor with the lowest health score
            lowest_rep = min(valid_reports, key=lambda r: (r.health_score, r.sensor_id))
            most_vulnerable = lowest_rep.sensor_id

            # Highest priority is the lowest score sensor among non-healthy sensors, or None if all healthy
            non_healthy = [r for r in valid_reports if r.health_status != "Healthy"]
            if non_healthy:
                prio_rep = min(non_healthy, key=lambda r: (r.health_score, r.sensor_id))
                highest_priority = prio_rep.sensor_id
            else:
                highest_priority = None
        else:
            overall_score = 0.0
            most_vulnerable = None
            highest_priority = None

        return NetworkHealthSummary(
            overall_network_health_score=overall_score,
            healthy_count=healthy_cnt,
            warning_count=warning_cnt,
            critical_count=critical_cnt,
            insufficient_data_count=insufficient_cnt,
            most_vulnerable_sensor=most_vulnerable,
            highest_priority_sensor=highest_priority,
            sensor_reports=reports,
            disclaimer=HEALTH_DISCLAIMER,
        )
