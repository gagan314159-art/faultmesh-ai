"""Live IoT Diagnostic Pipeline Validation Test Suite.

Comprehensive 16-point validation suite for FaultMesh AI 2.0 Live IoT Mode:
1. Normal readings from all five sensors.
2. Sustained abnormal pattern in S1.
3. Sustained abnormal pattern in S2.
4. Sustained abnormal pattern in S3.
5. Sustained abnormal pattern in S4.
6. Sustained abnormal pattern in S5.
7. Temporary noise affecting one sensor.
8. Delayed messages.
9. Duplicate messages.
10. Out-of-order messages.
11. Invalid and conflicting sensor IDs.
12. Incorrect measurement units.
13. Missing readings without zero imputation.
14. Insufficient calibration data.
15. Multiple correlated sensors.
16. Recovery to normal readings after an anomaly.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import pytest
import numpy as np
import pandas as pd

from faultmesh.simulator import SensorSimulator, DEFAULT_BASELINES
from faultmesh.ingestion.data_validator import DataValidator, SUPPORTED_SENSORS, STANDARD_UNITS
from faultmesh.ingestion.data_mapper import TelemetryPoint
from faultmesh.ingestion.buffer import LiveTelemetryManager, SensorBuffer
from faultmesh.processing import add_features
from faultmesh.detector import AnomalyDetector, first_onset
from faultmesh.explainer import FaultExplainer, InvestigationReport
from faultmesh.localizer import FaultLocalizer
from faultmesh.ingestion.mqtt_client import MQTTConnectionConfig
from dashboard.utils import SimConfig, run_simulation


def feed_stream(
    mgr: LiveTelemetryManager,
    df: pd.DataFrame,
    start_step: int = 0,
    base_epoch: float = 1700000000.0,
    step_duration: float = 1.0,
) -> None:
    """Helper to feed DataFrame rows as multi-sensor telemetry points into LiveTelemetryManager."""
    for idx, (_, row) in enumerate(df.iterrows()):
        step_idx = start_step + idx
        ep = base_epoch + (step_idx * step_duration)
        ts_iso = (datetime.fromtimestamp(ep, tz=timezone.utc)).isoformat()
        for s in SUPPORTED_SENSORS:
            pt = TelemetryPoint(
                machine_id="MACHINE_01",
                sensor_id=s,
                value=float(row[s]),
                unit=STANDARD_UNITS[s],
                timestamp=ts_iso,
                epoch=ep,
            )
            mgr.ingest_point(pt)


class TestLiveIoTDiagnosticPipeline:
    """Systematic 16-test suite verifying live IoT diagnostic logic and fault isolation."""

    # 1. Normal readings from all five sensors
    def test_1_normal_readings_all_five_sensors(self) -> None:
        """Verify nominal readings produce 'NORMAL' status with no primary suspect."""
        sim = SensorSimulator(seed=101)
        df_normal = sim.generate(n_steps=80, fault=None)

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        # Ingest first 40 for calibration
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        ok, msg = mgr.confirm_and_calibrate()
        assert ok is True
        assert mgr.calibration_status == "calibrated"

        # Ingest remaining 40 normal readings
        feed_stream(mgr, df_normal.iloc[40:], start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "NORMAL"
        assert res["primary_suspect"] is None
        assert "normal" in res["diagnosis_summary"].lower()

        diags = res["sensor_diagnostics"]
        for s in SUPPORTED_SENSORS:
            assert diags[s]["anomaly_status"] == "Normal"
            assert diags[s]["root_cause_status"] == "Normal"

    # 2. Sustained abnormal pattern in S1
    def test_2_sustained_abnormal_pattern_s1(self) -> None:
        """Verify sustained drift in S1 is accurately identified as primary root cause."""
        sim = SensorSimulator(seed=102)
        df_normal = sim.generate(n_steps=50, fault=None)
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S1", "type": "drift", "start": 0, "magnitude": 15.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S1"
        assert res["sensor_diagnostics"]["S1"]["root_cause_status"] == "Root-Cause Candidate"
        assert res["sensor_diagnostics"]["S1"]["anomaly_status"] == "Confirmed Sustained Anomaly"

    # 3. Sustained abnormal pattern in S2
    def test_3_sustained_abnormal_pattern_s2(self) -> None:
        """Verify sustained drift in S2 is accurately identified as primary root cause."""
        sim = SensorSimulator(seed=103)
        df_normal = sim.generate(n_steps=50, fault=None)
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S2", "type": "drift", "start": 0, "magnitude": 2.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S2"
        assert res["sensor_diagnostics"]["S2"]["root_cause_status"] == "Root-Cause Candidate"

    # 4. Sustained abnormal pattern in S3
    def test_4_sustained_abnormal_pattern_s3(self) -> None:
        """Verify sustained drift in S3 is accurately identified as primary root cause."""
        sim = SensorSimulator(seed=104)
        df_normal = sim.generate(n_steps=50, fault=None)
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S3", "type": "drift", "start": 0, "magnitude": 3.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S3"
        assert res["sensor_diagnostics"]["S3"]["root_cause_status"] == "Root-Cause Candidate"

    # 5. Sustained abnormal pattern in S4
    def test_5_sustained_abnormal_pattern_s4(self) -> None:
        """Verify sustained drift in S4 is accurately identified as primary root cause."""
        sim = SensorSimulator(seed=105)
        df_normal = sim.generate(n_steps=50, fault=None)
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S4", "type": "drift", "start": 0, "magnitude": 15.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S4"
        assert res["sensor_diagnostics"]["S4"]["root_cause_status"] == "Root-Cause Candidate"

    # 6. Sustained abnormal pattern in S5
    def test_6_sustained_abnormal_pattern_s5(self) -> None:
        """Verify sustained drift in S5 is accurately identified as primary root cause."""
        sim = SensorSimulator(seed=106)
        df_normal = sim.generate(n_steps=50, fault=None)
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S5", "type": "drift", "start": 0, "magnitude": 6.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S5"
        assert res["sensor_diagnostics"]["S5"]["root_cause_status"] == "Root-Cause Candidate"

    # 7. Temporary noise affecting one sensor
    def test_7_temporary_noise_affecting_one_sensor(self) -> None:
        """Verify a momentary single-reading spike does not trigger a confirmed fault onset."""
        sim = SensorSimulator(seed=107)
        df_normal = sim.generate(n_steps=60, fault=None)

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        # Ingest normal data with an isolated 1-sample noise glitch on S3 at step 45
        df_post = df_normal.iloc[40:].copy().reset_index(drop=True)
        df_post.loc[5, "S3"] = df_post.loc[5, "S3"] + 10.0  # isolated glitch

        feed_stream(mgr, df_post, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        # Since the spike was only 1 sample, min_consecutive=3 is not satisfied
        assert res["onsets"]["S3"] is None
        # Does not declare confirmed sustained fault
        assert res["fault_diagnosis_status"] in ("NORMAL", "UNCERTAIN")
        assert res["primary_suspect"] != "S3" or res["fault_diagnosis_status"] == "UNCERTAIN"

    # 8. Delayed messages
    def test_8_delayed_messages(self) -> None:
        """Verify delayed messages with slight subsecond jitter are properly clustered."""
        mgr = LiveTelemetryManager(min_calibration_samples=5, target_calibration_samples=10)
        base_epoch = 1700000000.0

        for i in range(10):
            # S1 arrives on time
            mgr.ingest_point(TelemetryPoint("M1", "S1", 60.1, "°C", "ts", base_epoch + i))
            # S2 arrives +0.1s delayed
            mgr.ingest_point(TelemetryPoint("M1", "S2", 2.01, "mm/s", "ts", base_epoch + i + 0.1))
            # S3 arrives +0.2s delayed
            mgr.ingest_point(TelemetryPoint("M1", "S3", 5.02, "bar", "ts", base_epoch + i + 0.2))
            # S4 arrives +0.15s delayed
            mgr.ingest_point(TelemetryPoint("M1", "S4", 30.1, "L/min", "ts", base_epoch + i + 0.15))
            # S5 arrives +0.3s delayed
            mgr.ingest_point(TelemetryPoint("M1", "S5", 10.05, "A", "ts", base_epoch + i + 0.3))

        synced = mgr.get_synchronized_dataframe()
        assert len(synced) == 10
        # All columns present
        for s in SUPPORTED_SENSORS:
            assert s in synced.columns

    # 9. Duplicate messages
    def test_9_duplicate_messages(self) -> None:
        """Verify duplicate messages with matching epoch are filtered and counted."""
        buf = SensorBuffer("S1", max_size=50)
        pt1 = TelemetryPoint("M1", "S1", 60.5, "°C", "2026-11-22T00:00:00Z", 1700000000.0)
        pt2 = TelemetryPoint("M1", "S1", 60.5, "°C", "2026-11-22T00:00:00Z", 1700000000.0)

        assert buf.add_point(pt1) is True
        assert buf.add_point(pt2) is False
        assert buf.duplicate_count == 1
        assert len(buf.points) == 1

    # 10. Out-of-order messages
    def test_10_out_of_order_messages(self) -> None:
        """Verify out-of-order messages maintain sorted chronological order."""
        buf = SensorBuffer("S1", max_size=50)
        pt1 = TelemetryPoint("M1", "S1", 60.1, "°C", "ts", 1700000001.0)
        pt2 = TelemetryPoint("M1", "S1", 60.3, "°C", "ts", 1700000003.0)
        pt_late = TelemetryPoint("M1", "S1", 60.2, "°C", "ts", 1700000002.0)

        buf.add_point(pt1)
        buf.add_point(pt2)
        # Ingest late arriving point
        accepted = buf.add_point(pt_late)
        assert accepted is True
        assert buf.out_of_order_count == 1

        pts = buf.get_points()
        assert len(pts) == 3
        # Chronological order verified
        assert pts[0].epoch < pts[1].epoch < pts[2].epoch
        assert pts[1].value == 60.2

    # 11. Invalid and conflicting sensor IDs
    def test_11_invalid_and_conflicting_sensor_ids(self) -> None:
        """Verify validator rejects invalid IDs and detects payload/topic conflicts."""
        validator = DataValidator()

        # Invalid sensor ID
        res_inv = validator.validate_message('{"sensor_id":"S9","value":50.0,"unit":"°C"}')
        assert res_inv.is_valid is False
        assert res_inv.error_type == "UNSUPPORTED_SENSOR"

        # Conflicting sensor ID: payload says S1, topic says S2
        res_conf = validator.validate_message(
            '{"sensor_id":"S1","value":60.0,"unit":"°C"}',
            topic="faultmesh/sensors/S2",
        )
        assert res_conf.is_valid is False
        assert res_conf.error_type == "CONFLICTING_SENSOR_ID"
        assert "conflicts" in res_conf.error.lower()

    # 12. Incorrect measurement units
    def test_12_incorrect_units(self) -> None:
        """Verify validator rejects mismatched engineering units."""
        validator = DataValidator()

        # S1 expected temperature °C, got pressure bar
        res = validator.validate_message('{"sensor_id":"S1","value":60.0,"unit":"bar"}')
        assert res.is_valid is False
        assert res.error_type == "INVALID_UNIT"

        # S3 expected pressure bar, got amperes A
        res3 = validator.validate_message('{"sensor_id":"S3","value":5.0,"unit":"A"}')
        assert res3.is_valid is False
        assert res3.error_type == "INVALID_UNIT"

    # 13. Missing readings without zero imputation
    def test_13_missing_readings_no_zero_imputation(self) -> None:
        """Verify incomplete multi-sensor time slices are not fabricated or zero-imputed."""
        mgr = LiveTelemetryManager(min_calibration_samples=5, target_calibration_samples=10)
        base_epoch = 1700000000.0

        # Feed 3 complete samples
        for i in range(3):
            for s in SUPPORTED_SENSORS:
                mgr.ingest_point(TelemetryPoint("M1", s, 50.0, STANDARD_UNITS[s], "ts", base_epoch + i))

        # At i=3, S4 is missing
        for s in ["S1", "S2", "S3", "S5"]:
            mgr.ingest_point(TelemetryPoint("M1", s, 50.0, STANDARD_UNITS[s], "ts", base_epoch + 3))

        synced = mgr.get_synchronized_dataframe()
        # Only 3 complete synchronized rows assembled (missing S4 not imputed as 0.0)
        assert len(synced) == 3

    # 14. Insufficient calibration data
    def test_14_insufficient_calibration_data(self) -> None:
        """Verify calibration requires minimum data and suspends AI analysis beforehand."""
        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=50)

        # Ingest only 10 samples
        base_epoch = 1700000000.0
        for i in range(10):
            for s in SUPPORTED_SENSORS:
                mgr.ingest_point(TelemetryPoint("M1", s, 50.0, STANDARD_UNITS[s], "ts", base_epoch + i))

        can_cal, count, msg = mgr.can_calibrate()
        assert can_cal is False
        assert count == 10

        ok, c_msg = mgr.confirm_and_calibrate()
        assert ok is False
        assert "insufficient" in c_msg.lower()

        # Analysis is suspended
        res = mgr.process_live_analysis()
        assert res["status"] == "awaiting_calibration"

    # 15. Multiple correlated sensors
    def test_15_multiple_correlated_sensors(self) -> None:
        """Verify initiating sensor is prioritized over delayed coupled recipient."""
        sim = SensorSimulator(seed=115)
        df_normal = sim.generate(n_steps=50, fault=None)
        # S1 drift fault physically couples to neighbor S3 in simulator
        df_fault = sim.generate(n_steps=50, fault={"sensor": "S1", "type": "drift", "start": 0, "magnitude": 25.0})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        feed_stream(mgr, df_fault, start_step=40)
        res = mgr.process_live_analysis()

        assert res["status"] == "success"
        # Primary suspect must be S1 (the initiator), not coupled S3
        assert res["primary_suspect"] == "S1"
        assert res["rankings"][0][0] == "S1"

        diags = res["sensor_diagnostics"]
        assert diags["S1"]["root_cause_status"] == "Root-Cause Candidate"
        # If S3 experienced coupled propagation, it is classified as Coupled Recipient, not primary
        if diags["S3"]["anomaly_status"] == "Confirmed Sustained Anomaly":
            assert diags["S3"]["root_cause_status"] == "Coupled Recipient"

    # 16. Recovery to normal readings after an anomaly
    def test_16_recovery_to_normal_readings_after_anomaly(self) -> None:
        """Verify sensor score subsides back to nominal after a transient abnormal pattern ends."""
        sim = SensorSimulator(seed=116)
        df_normal = sim.generate(n_steps=50, fault=None)
        # Brief spike fault
        df_spike = sim.generate(n_steps=15, fault={"sensor": "S2", "type": "spike", "start": 0, "magnitude": 3.0, "duration": 5})
        # Subsequent normal recovery stream
        df_recovery = sim.generate(n_steps=40, fault=None)

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=40)
        feed_stream(mgr, df_normal.iloc[:40], start_step=0)
        mgr.confirm_and_calibrate()

        # Feed spike fault
        feed_stream(mgr, df_spike, start_step=40)
        # Feed recovery normal stream
        feed_stream(mgr, df_recovery, start_step=55)

        res = mgr.process_live_analysis()
        assert res["status"] == "success"

        # In the tail 30 samples, sensor readings have stabilized back to baseline
        recent_s2_score = res["scores"]["S2"].iloc[-15:].mean()
        assert recent_s2_score < 0.50


class TestLiveIoTAnomalyDetectionAndExplainability:
    """Targeted validation for live IoT anomaly detection, explainable investigation, and UI integrity."""

    # 1. Live MQTT telemetry reaching the anomaly detector
    def test_1_live_telemetry_reaches_anomaly_detector(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=20, target_calibration_samples=30)
        assert mgr.calibrated_detector is not None

        sim = SensorSimulator(seed=201)
        df_normal = sim.generate(n_steps=30, fault=None)
        feed_stream(mgr, df_normal, start_step=0)
        mgr.confirm_and_calibrate()

        res = mgr.process_live_analysis()
        assert res["status"] == "success"
        assert "scores" in res
        assert all(s in res["scores"].columns for s in ["S1", "S2", "S3", "S4", "S5"])
        assert len(res["scores"]) == 30

    # 2. Independent anomaly scoring for all five sensors
    def test_2_independent_anomaly_scoring_for_all_five_sensors(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=25, target_calibration_samples=30)
        sim = SensorSimulator(seed=202)
        df_normal = sim.generate(n_steps=30, fault=None)
        feed_stream(mgr, df_normal, start_step=0)
        mgr.confirm_and_calibrate()

        # Ingest severe drift on S4 specifically
        df_fault = sim.generate(n_steps=20, fault={"sensor": "S4", "type": "drift", "start": 0, "magnitude": 50.0})
        feed_stream(mgr, df_fault, start_step=30)

        res = mgr.process_live_analysis()
        scores = res["scores"]
        # S4 anomaly score should be high
        assert scores["S4"].iloc[-1] > 0.60
        # Other sensors must have distinct independent anomaly calculations
        s4_score = scores["S4"].iloc[-1]
        s1_score = scores["S1"].iloc[-1]
        assert s4_score != s1_score

    # 3. Detection of a sustained abnormal pattern
    def test_3_detection_of_sustained_abnormal_pattern(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=25, target_calibration_samples=30)
        sim = SensorSimulator(seed=203)
        df_normal = sim.generate(n_steps=30, fault=None)
        feed_stream(mgr, df_normal, start_step=0)
        mgr.confirm_and_calibrate()

        # Feed sustained drift fault on S3
        df_fault = sim.generate(n_steps=40, fault={"sensor": "S3", "type": "drift", "start": 0, "magnitude": 15.0})
        feed_stream(mgr, df_fault, start_step=30)

        res = mgr.process_live_analysis()
        assert res["onsets"]["S3"] is not None
        assert res["fault_diagnosis_status"] == "CONFIRMED"
        assert res["primary_suspect"] == "S3"

    # 4. Rejection or handling of invalid and insufficient telemetry
    def test_4_rejection_or_handling_of_invalid_and_insufficient_telemetry(self) -> None:
        validator = DataValidator()
        # Invalid sensor ID
        res1 = validator.validate_message({"sensor_id": "UNKNOWN", "value": 10.0, "timestamp": "2026-11-21T00:00:00Z"})
        assert res1.is_valid is False
        assert res1.error_type == "UNSUPPORTED_SENSOR"

        # Invalid unit
        res2 = validator.validate_message({"sensor_id": "S1", "value": 60.0, "unit": "psi", "timestamp": "2026-11-21T00:00:00Z"})
        assert res2.is_valid is False
        assert res2.error_type == "INVALID_UNIT"

        # Insufficient telemetry handling in manager
        mgr = LiveTelemetryManager(min_calibration_samples=30)
        assert mgr.can_calibrate()[0] is False

    # 5. Correct detection onset calculation
    def test_5_correct_detection_onset_calculation(self) -> None:
        scores_df = pd.DataFrame({
            "time": list(range(10)),
            "S1": [0.2, 0.2, 0.7, 0.8, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2],  # 3 consecutive: t=2, 3, 4 -> onset t=2
            "S2": [0.2, 0.7, 0.2, 0.8, 0.2, 0.7, 0.2, 0.2, 0.2, 0.2],  # isolated spikes -> no onset
        })
        onsets = first_onset(scores_df, threshold=0.60, min_consecutive=3)
        assert onsets["S1"] == 2
        assert onsets["S2"] is None

    # 6. Explainable Fault Investigation using live diagnostic evidence
    def test_6_explainable_investigation_using_live_diagnostic_evidence(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=25, target_calibration_samples=30)
        sim = SensorSimulator(seed=206)
        df_normal = sim.generate(n_steps=30, fault=None)
        feed_stream(mgr, df_normal, start_step=0)
        mgr.confirm_and_calibrate()

        df_fault = sim.generate(n_steps=20, fault={"sensor": "S2", "type": "drift", "start": 0, "magnitude": 10.0})
        feed_stream(mgr, df_fault, start_step=30)

        res = mgr.process_live_analysis()
        rep = res["investigation_report"]
        assert isinstance(rep, InvestigationReport)
        assert rep.suspected_sensor == "S2"
        assert len(rep.evidence) >= 3
        assert len(rep.timing_comparisons) == 5
        assert rep.confidence_category in ("Strong", "Moderate")

    # 7. Consistency between anomaly results and investigation explanations
    def test_7_consistency_between_anomaly_results_and_investigation_explanations(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=25, target_calibration_samples=30)
        sim = SensorSimulator(seed=207)
        df_normal = sim.generate(n_steps=30, fault=None)
        feed_stream(mgr, df_normal, start_step=0)
        mgr.confirm_and_calibrate()

        df_fault = sim.generate(n_steps=20, fault={"sensor": "S1", "type": "drift", "start": 0, "magnitude": 30.0})
        feed_stream(mgr, df_fault, start_step=30)

        res = mgr.process_live_analysis()
        rep = res["investigation_report"]
        assert rep.suspected_sensor == res["primary_suspect"]
        assert rep.localization_score == round(res["rankings"][0][1], 4)

    # 8. Root-cause uncertainty when evidence is insufficient
    def test_8_root_cause_uncertainty_when_evidence_is_insufficient(self) -> None:
        explainer = FaultExplainer()
        # Uncertain diagnosis status
        rep = explainer.explain(
            scores_df=None,
            onsets={"S1": None, "S2": None, "S3": None, "S4": None, "S5": None},
            rankings=[("S3", 0.65, {"own_anomaly": 0.65, "detection_earliness": 0.0, "neighbour_support": 0.0, "coverage": 0.2})],
            anomaly_scores={"S3": 0.65},
            diagnosis_status="UNCERTAIN",
        )
        assert rep.suspected_sensor is None
        assert rep.confidence_category == "Limited"
        assert "UNCERTAIN" in rep.investigation_summary

    # 9. Absence of the Fault Target Sensor UI control
    def test_9_absence_of_the_fault_target_sensor_ui_control(self) -> None:
        import inspect
        from dashboard import layout
        src = inspect.getsource(layout.sidebar_live_iot_controls)
        # Verify Fault Target Sensor selector is completely absent from Live IoT controls
        assert "Fault Target Sensor" not in src
        assert "selectbox(\"Fault Target Sensor\"" not in src

    # 10. Correct operation after Streamlit reruns
    def test_10_correct_operation_after_reruns(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=20, target_calibration_samples=25)
        sim = SensorSimulator(seed=210)
        df = sim.generate(n_steps=25, fault=None)
        feed_stream(mgr, df, start_step=0)
        mgr.confirm_and_calibrate()

        # Simulate 5 consecutive rerun cycles
        for cycle in range(5):
            res = mgr.process_live_analysis()
            assert res["status"] == "success"
            assert res["fault_diagnosis_status"] == "NORMAL"
            assert len(res["sensor_diagnostics"]) == 5

    # 11. Preservation of MQTT connectivity
    def test_11_preservation_of_mqtt_connectivity(self) -> None:
        config = MQTTConnectionConfig(
            host="broker.emqx.io",
            port=8883,
            topic="faultmesh/sensors/#",
            username="test_user",
            password="test_password",
            use_tls=True,
        )
        assert config.host == "broker.emqx.io"
        assert config.port == 8883
        assert config.use_tls is True
        assert config.username == "test_user"

    # 12. Preservation of existing Simulation Mode functionality
    def test_12_preservation_of_existing_simulation_mode_functionality(self) -> None:
        cfg = SimConfig(
            total_steps=200,
            fault_sensor="S3",
            fault_type="drift",
            fault_start=100,
            fault_magnitude_ratio=0.5,
            seed=42,
        )
        readings, fault_dict = run_simulation(cfg, use_cache=False)
        assert len(readings) == 200
        assert "S3" in readings.columns
        assert fault_dict is not None
        assert fault_dict.get("sensor") == "S3"
