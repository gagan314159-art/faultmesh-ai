"""Rolling buffer and live telemetry management module for FaultMesh AI.

Responsibility:
    Maintains chronological rolling time-series buffers for sensors S1 through S5,
    detects stale or disconnected channels, handles duplicate and out-of-order packets,
    assembles synchronized multi-sensor time slices without fabricating data,
    and manages the baseline calibration lifecycle for live AI fault analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from collections import deque
from datetime import datetime, timezone
import time
import threading
import pandas as pd
import numpy as np

from faultmesh.ingestion.data_validator import SUPPORTED_SENSORS, STANDARD_UNITS
from faultmesh.ingestion.data_mapper import TelemetryPoint, DataMapper
from faultmesh.detector import AnomalyDetector, first_onset
from faultmesh.processing import add_features
from faultmesh.localizer import FaultLocalizer
from faultmesh.graph import SensorNetworkGraph
from faultmesh.explainer import FaultExplainer, InvestigationReport
from faultmesh.health import SensorHealthMonitor, SensorHealthReport, NetworkHealthSummary
from faultmesh.maintenance import MaintenanceAdvisor, MaintenanceAdvisory, NetworkMaintenanceSummary
from faultmesh.reporting import build_investigation_summary


class SensorBuffer:
    """Individual sensor ring-buffer maintaining chronological order and deduplication."""

    def __init__(self, sensor_id: str, max_size: int = 500) -> None:
        self.sensor_id = sensor_id
        self.max_size = max_size
        self.points: List[TelemetryPoint] = []
        self._seen_epochs: Set[float] = set()
        self.last_received_time: Optional[float] = None
        self.total_points: int = 0
        self.duplicate_count: int = 0
        self.out_of_order_count: int = 0

    def add_point(self, point: TelemetryPoint) -> bool:
        """Add a telemetry point maintaining chronological order and filtering duplicates.

        Returns:
            True if point was accepted, False if rejected as duplicate.
        """
        # Deduplication check by rounded epoch
        epoch_key = round(point.epoch, 3)
        if epoch_key in self._seen_epochs:
            self.duplicate_count += 1
            return False

        now = time.time()
        self.last_received_time = now
        self.total_points += 1
        self._seen_epochs.add(epoch_key)

        # Check if out of order
        if self.points and point.epoch < self.points[-1].epoch:
            self.out_of_order_count += 1
            # Insert in sorted order
            idx = 0
            while idx < len(self.points) and self.points[idx].epoch < point.epoch:
                idx += 1
            self.points.insert(idx, point)
        else:
            self.points.append(point)

        # Maintain max size
        if len(self.points) > self.max_size:
            discarded = self.points.pop(0)
            self._seen_epochs.discard(round(discarded.epoch, 3))

        return True

    def get_points(self) -> List[TelemetryPoint]:
        """Return shallow copy of chronological telemetry points."""
        return list(self.points)

    def is_stale(self, threshold_seconds: float = 10.0, current_time: Optional[float] = None) -> bool:
        """Check if sensor hasn't received new data within threshold_seconds."""
        if self.last_received_time is None:
            return True
        now = current_time or time.time()
        return (now - self.last_received_time) > threshold_seconds

    def clear(self) -> None:
        """Clear buffer contents."""
        self.points.clear()
        self._seen_epochs.clear()
        self.last_received_time = None
        self.total_points = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0


class LiveTelemetryManager:
    """Manages multi-sensor rolling buffers, baseline calibration, and live analysis execution."""

    def __init__(
        self,
        max_buffer_size: int = 500,
        stale_threshold_seconds: float = 10.0,
        min_calibration_samples: int = 30,
        target_calibration_samples: int = 100,
    ) -> None:
        """Initialize telemetry manager with sensor buffers and calibration state."""
        self.lock = threading.Lock()
        self.max_buffer_size = max_buffer_size
        self.stale_threshold = stale_threshold_seconds
        self.min_calibration_samples = min_calibration_samples
        self.target_calibration_samples = target_calibration_samples

        self.buffers: Dict[str, SensorBuffer] = {
            s: SensorBuffer(s, max_size=max_buffer_size) for s in SUPPORTED_SENSORS
        }

        # Synchronized row buffer
        self.synchronized_rows: List[Dict[str, Any]] = []

        # Calibration State
        self.calibration_status: str = "uncalibrated"  # "uncalibrated", "calibrating", "calibrated"
        self.calibration_samples_collected: int = 0
        self.calibrated_at: Optional[str] = None
        self.calibrated_detector: Optional[AnomalyDetector] = None
        self.baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.operator_confirmed: bool = False

        # Latest AI Results Cache
        self.latest_readings_df: Optional[pd.DataFrame] = None
        self.latest_scores_df: Optional[pd.DataFrame] = None
        self.latest_onsets: Optional[Dict[str, Optional[int]]] = None
        self.latest_rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None
        self.latest_investigation: Optional[InvestigationReport] = None
        self.latest_health_reports: Optional[Dict[str, Any]] = None
        self.latest_network_health: Optional[NetworkHealthSummary] = None
        self.latest_advisories: Optional[Dict[str, Any]] = None
        self.latest_maintenance_summary: Optional[NetworkMaintenanceSummary] = None
        self.latest_executive_summary: Optional[Dict[str, Any]] = None

    def ingest_point(self, point: TelemetryPoint) -> bool:
        """Ingest a validated telemetry point into the appropriate sensor buffer.

        Returns:
            True if accepted, False if duplicate or rejected.
        """
        sensor_id = point.sensor_id
        if sensor_id not in self.buffers:
            return False

        with self.lock:
            accepted = self.buffers[sensor_id].add_point(point)
            if accepted:
                self._attempt_row_synchronization()
            return accepted

    def _attempt_row_synchronization(self) -> None:
        """Assemble synchronized multi-sensor observation rows when all 5 sensors are populated.

        Strict rule: Never fabricate or impute missing sensor readings.
        """
        # Determine how many synchronized samples can be formed
        lengths = [len(self.buffers[s].points) for s in SUPPORTED_SENSORS]
        min_len = min(lengths) if lengths else 0

        # We keep synchronized_rows aligned with min_len
        current_synced = len(self.synchronized_rows)
        if min_len > current_synced:
            for idx in range(current_synced, min_len):
                row = {
                    "time": idx,
                    "_timestamp": self.buffers["S1"].points[idx].timestamp,
                    "_epoch": self.buffers["S1"].points[idx].epoch,
                }
                for s in SUPPORTED_SENSORS:
                    row[s] = self.buffers[s].points[idx].value
                self.synchronized_rows.append(row)

            # Cap synchronized rows to max_buffer_size
            if len(self.synchronized_rows) > self.max_buffer_size:
                excess = len(self.synchronized_rows) - self.max_buffer_size
                self.synchronized_rows = self.synchronized_rows[excess:]
                # Re-index time
                for i, r in enumerate(self.synchronized_rows):
                    r["time"] = i

    def get_latest_readings(self) -> Dict[str, Dict[str, Any]]:
        """Return latest reading snapshot, unit, timestamp, and staleness for each sensor."""
        now = time.time()
        results: Dict[str, Dict[str, Any]] = {}

        with self.lock:
            for s in SUPPORTED_SENSORS:
                buf = self.buffers[s]
                pts = buf.points
                if pts:
                    latest_pt = pts[-1]
                    stale = buf.is_stale(self.stale_threshold, now)
                    results[s] = {
                        "sensor_id": s,
                        "value": latest_pt.value,
                        "unit": latest_pt.unit,
                        "timestamp": latest_pt.timestamp,
                        "epoch": latest_pt.epoch,
                        "stale": stale,
                        "status": "Stale / Offline" if stale else "Active",
                        "point_count": len(pts),
                        "duplicates": buf.duplicate_count,
                        "out_of_order": buf.out_of_order_count,
                    }
                else:
                    results[s] = {
                        "sensor_id": s,
                        "value": None,
                        "unit": STANDARD_UNITS.get(s, ""),
                        "timestamp": "No data",
                        "epoch": 0.0,
                        "stale": True,
                        "status": "Disconnected",
                        "point_count": 0,
                        "duplicates": 0,
                        "out_of_order": 0,
                    }
        return results

    def get_synchronized_dataframe(self) -> pd.DataFrame:
        """Return synchronized DataFrame of all 5 sensors, or empty DataFrame if incomplete."""
        with self.lock:
            if not self.synchronized_rows:
                return pd.DataFrame(columns=["time"] + SUPPORTED_SENSORS)
            return pd.DataFrame(self.synchronized_rows)

    def can_calibrate(self) -> Tuple[bool, int, str]:
        """Check whether enough synchronized readings are available to perform baseline calibration.

        Returns:
            Tuple of (can_calibrate, available_samples, reason_message)
        """
        with self.lock:
            count = len(self.synchronized_rows)
            if count < self.min_calibration_samples:
                return False, count, f"Collecting baseline: {count}/{self.min_calibration_samples} required samples"
            return True, count, f"Ready for calibration: {count} samples available (minimum {self.min_calibration_samples})"

    def confirm_and_calibrate(self) -> Tuple[bool, str]:
        """Authorized operator confirmation to train baseline model on verified normal data.

        Returns:
            Tuple of (success, message).
        """
        with self.lock:
            df = pd.DataFrame(self.synchronized_rows) if self.synchronized_rows else pd.DataFrame()
            if len(df) < self.min_calibration_samples:
                return False, f"Insufficient baseline samples: {len(df)}/{self.min_calibration_samples} required."

            # Verify that all 5 sensor columns exist and have non-zero variance
            for s in SUPPORTED_SENSORS:
                if s not in df.columns or df[s].isna().all():
                    return False, f"Sensor {s} has missing telemetry; cannot calibrate baseline."

            # Use up to target_calibration_samples for calibration training
            train_slice = df.iloc[: min(len(df), self.target_calibration_samples)].copy()
            train_feats = add_features(train_slice)

            # Fit Isolation Forest detector
            detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
            detector.fit(train_feats)

            self.calibrated_detector = detector
            self.calibration_status = "calibrated"
            self.operator_confirmed = True
            self.calibrated_at = datetime.now(timezone.utc).isoformat()
            self.calibration_samples_collected = len(train_slice)

            return True, f"Baseline successfully calibrated on {len(train_slice)} confirmed normal samples."

    def reset_calibration(self) -> None:
        """Reset calibration status requiring operator re-confirmation."""
        with self.lock:
            self.calibration_status = "uncalibrated"
            self.operator_confirmed = False
            self.calibrated_at = None
            self.calibrated_detector = None
            self.latest_scores_df = None
            self.latest_onsets = None
            self.latest_rankings = None
            self.latest_investigation = None
            self.latest_health_reports = None
            self.latest_network_health = None
            self.latest_advisories = None
            self.latest_maintenance_summary = None
            self.latest_executive_summary = None

    def process_live_analysis(
        self,
        min_eval_window: int = 15,
        benchmark: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full FaultMesh AI analytical pipeline if baseline is calibrated.

        Returns:
            Dictionary containing latest pipeline results, or status indicator if uncalibrated.
        """
        with self.lock:
            if not self.operator_confirmed or self.calibrated_detector is None:
                return {
                    "status": "awaiting_calibration",
                    "message": "AI analysis suspended. Telemetry baseline calibration required.",
                    "calibration_status": self.calibration_status,
                    "samples_available": len(self.synchronized_rows),
                }

            if len(self.synchronized_rows) < min_eval_window:
                return {
                    "status": "window_too_small",
                    "message": f"Need at least {min_eval_window} synchronized readings for evaluation (have {len(self.synchronized_rows)}).",
                    "calibration_status": self.calibration_status,
                }

            df = pd.DataFrame(self.synchronized_rows)
            feats_df = add_features(df)

            # 1. Anomaly scoring using the calibrated detector
            scores_df = self.calibrated_detector.score(feats_df)
            onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

            # 2. Root-cause localization
            sensor_cols = [s for s in SUPPORTED_SENSORS if s in scores_df.columns]
            n_steps = len(scores_df)
            tail_slice = scores_df.iloc[-min(50, n_steps):]
            is_transient = any(
                float(scores_df[s].max()) >= 0.6 and float(tail_slice[s].mean()) < 0.4
                for s in sensor_cols
            )
            if is_transient:
                anomaly_scores = {s: float(scores_df[s].max()) for s in sensor_cols}
            else:
                anomaly_scores = {s: float(tail_slice[s].mean()) for s in sensor_cols}

            localizer = FaultLocalizer()
            rankings = localizer.rank_suspects(anomaly_scores, onsets, total_steps=n_steps)

            # 3. Explainability
            graph = SensorNetworkGraph()
            explainer = FaultExplainer(network_graph=graph)
            inv_report = explainer.explain(
                scores_df=scores_df,
                onsets=onsets,
                rankings=rankings,
                anomaly_scores=anomaly_scores,
            )

            # 4. Sensor Health Monitoring
            monitor = SensorHealthMonitor()
            health_records = monitor.assess_all(df, scores_df, onsets=onsets)
            net_health = monitor.summarize_network_health(health_records)

            # 5. Predictive Maintenance
            advisor = MaintenanceAdvisor()
            advisories = advisor.advise_fleet(
                health_reports=health_records,
                rankings=rankings,
                investigation_report=inv_report,
                scores_df=scores_df,
                onsets=onsets,
            )
            maint_summary = advisor.summarize_network_maintenance(advisories)

            # 6. Executive Reporting Summary
            exec_summary = build_investigation_summary(
                cfg_info={"mode": "Live IoT Telemetry", "calibrated_at": self.calibrated_at},
                readings_df=df,
                scores_df=scores_df,
                onsets=onsets,
                rankings=rankings,
                investigation_report=inv_report,
                health_reports=health_records,
                advisories=advisories,
                network_health=net_health,
                maintenance_summary=maint_summary,
                benchmark=benchmark,
            )

            # Cache results
            self.latest_readings_df = df
            self.latest_scores_df = scores_df
            self.latest_onsets = onsets
            self.latest_rankings = rankings
            self.latest_investigation = inv_report
            self.latest_health_reports = health_records
            self.latest_network_health = net_health
            self.latest_advisories = advisories
            self.latest_maintenance_summary = maint_summary
            self.latest_executive_summary = exec_summary

            return {
                "status": "success",
                "calibration_status": "calibrated",
                "readings": df,
                "scores": scores_df,
                "onsets": onsets,
                "rankings": rankings,
                "investigation_report": inv_report,
                "health_reports": health_records,
                "network_health": net_health,
                "advisories": advisories,
                "maintenance_summary": maint_summary,
                "executive_summary": exec_summary,
            }

    def clear_all(self) -> None:
        """Clear all buffers, synchronized rows, and analysis caches."""
        with self.lock:
            for b in self.buffers.values():
                b.clear()
            self.synchronized_rows.clear()
            self.reset_calibration()
