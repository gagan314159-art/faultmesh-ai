"""Comprehensive unit and scenario tests for Sensor Health Monitoring Engine (faultmesh/health.py).

Verifies:
1. All five sensors evaluation.
2. Completely normal simulation.
3. Drift fault health impact.
4. Spike fault health impact.
5. Stuck fault health impact (variance loss).
6. Healthy, Warning, and Critical classifications.
7. Missing readings handling.
8. Insufficient data handling.
9. Network health aggregation.
10. Deterministic results.
11. Changing simulation parameters.
12. Invalid or empty input handling.
"""

import pytest
import pandas as pd
import numpy as np

from faultmesh.health import (
    SensorHealthMonitor,
    SensorHealthRecord,
    NetworkHealthSummary,
    DEFAULT_BASELINES,
)
from faultmesh.simulator import SensorSimulator
from faultmesh.detector import AnomalyDetector, first_onset


def test_1_all_five_sensors_evaluated():
    """Verify health monitor evaluates all five sensors in default network."""
    sim = SensorSimulator(seed=42)
    df = sim.generate(n_steps=150)
    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(df)

    assert len(reports) == 5
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        assert s in reports
        rep = reports[s]
        assert isinstance(rep, SensorHealthRecord)
        assert rep.sensor_id == s
        assert 0.0 <= rep.health_score <= 100.0
        assert rep.health_status in ("Healthy", "Warning", "Critical")


def test_2_completely_normal_simulation():
    """Verify completely normal simulation produces Healthy status for all sensors."""
    sim = SensorSimulator(seed=42)
    # Generate clean normal stream with no faults
    df = sim.generate(n_steps=200, fault=None)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(df, scores_df, onsets=onsets)
    summary = monitor.summarize_network_health(reports)

    # In clean normal operation, all sensors should be Healthy
    for s, rep in reports.items():
        assert rep.health_status == "Healthy", f"Sensor {s} was {rep.health_status} in clean stream"
        assert rep.health_score >= 85.0
        assert rep.recommended_monitoring_priority == "Normal"

    assert summary.healthy_count == 5
    assert summary.critical_count == 0
    assert summary.warning_count == 0
    assert summary.overall_network_health_score >= 85.0
    assert summary.highest_priority_sensor is None


def test_3_drift_fault_health_impact():
    """Verify continuous drift fault drives the target sensor into Warning or Critical condition."""
    sim = SensorSimulator(seed=42)
    fault_cfg = {"sensor": "S1", "type": "drift", "start": 120, "magnitude": 30.0}
    df = sim.generate(n_steps=250, fault=fault_cfg)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(df, scores_df, onsets=onsets)

    s1_rep = reports["S1"]
    assert s1_rep.health_status in ("Critical", "Warning")
    assert s1_rep.health_score < 85.0
    assert s1_rep.anomaly_frequency > 0.20
    assert s1_rep.maximum_anomaly_score >= 0.70
    assert s1_rep.detection_onset is not None
    assert any("offset" in f.lower() or "deviation" in f.lower() for f in s1_rep.degradation_factors)


def test_4_spike_fault_health_impact():
    """Verify transient spike fault registers peak anomaly and appropriate status."""
    sim = SensorSimulator(seed=42)
    fault_cfg = {"sensor": "S2", "type": "spike", "start": 140, "magnitude": 1.5, "duration": 10}
    df = sim.generate(n_steps=200, fault=fault_cfg)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(df, scores_df, onsets=onsets)

    s2_rep = reports["S2"]
    # Peak score should reflect the transient spike
    assert s2_rep.maximum_anomaly_score >= 0.60
    assert s2_rep.health_score < 100.0


def test_5_stuck_fault_health_impact():
    """Verify frozen sensor triggers zero-variance detection and Critical/Warning status."""
    sim = SensorSimulator(seed=42)
    fault_cfg = {"sensor": "S3", "type": "stuck", "start": 130, "magnitude": 2.5}
    df = sim.generate(n_steps=220, fault=fault_cfg)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(df, scores_df, onsets=onsets)

    s3_rep = reports["S3"]
    assert s3_rep.health_status in ("Critical", "Warning")
    assert any("freeze" in f.lower() or "variance" in f.lower() for f in s3_rep.degradation_factors)


def test_6_healthy_warning_critical_classifications():
    """Verify score boundaries produce exact Healthy, Warning, and Critical classifications."""
    monitor = SensorHealthMonitor(healthy_threshold=85.0, warning_threshold=65.0)

    # 1. Healthy: readings with normal slight variance near baseline (60.0), scores low
    np.random.seed(42)
    h_readings = pd.Series(60.0 + np.random.normal(0, 0.1, 50))
    h_scores = pd.Series([0.1] * 50)
    h_rep = monitor.assess_sensor("S1", h_readings, h_scores)
    assert h_rep.health_status == "Healthy"
    assert h_rep.health_score >= 85.0

    # 2. Warning: moderate baseline shift (~2.5 sigma) with mild anomaly rate and normal variance
    np.random.seed(42)
    w_readings = pd.Series(
        [60.0 + x for x in np.random.normal(0, 0.15, 35)]
        + [62.5 + x for x in np.random.normal(0, 0.15, 15)]
    )
    w_scores = pd.Series([0.1] * 35 + [0.65] * 15)
    w_rep = monitor.assess_sensor("S1", w_readings, w_scores)
    assert w_rep.health_status == "Warning"
    assert 65.0 <= w_rep.health_score < 85.0

    # 3. Critical: massive shift (>10 sigma) and high anomaly frequency
    np.random.seed(42)
    c_readings = pd.Series(
        [60.0 + x for x in np.random.normal(0, 0.15, 20)]
        + [90.0 + x for x in np.random.normal(0, 0.15, 30)]
    )
    c_scores = pd.Series([0.1] * 20 + [0.95] * 30)
    c_rep = monitor.assess_sensor("S1", c_readings, c_scores)
    assert c_rep.health_status == "Critical"
    assert c_rep.health_score < 65.0


def test_7_missing_readings_handling():
    """Verify handling when an individual sensor column is missing from readings DataFrame."""
    # DataFrame missing S4
    partial_df = pd.DataFrame({
        "time": list(range(50)),
        "S1": [60.0] * 50,
        "S2": [2.0] * 50,
        "S3": [5.0] * 50,
        "S5": [10.0] * 50,
    })
    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(partial_df)

    assert "S4" in reports
    assert reports["S4"].health_status == "Insufficient Data"
    assert reports["S4"].health_score == 0.0
    assert reports["S4"].recommended_monitoring_priority == "Verify Telemetry"
    assert "not found" in reports["S4"].health_explanation.lower()


def test_8_insufficient_data_handling():
    """Verify fewer than min_samples_required produces Insufficient Data status."""
    monitor = SensorHealthMonitor(min_samples_required=10)
    # Only 5 samples
    short_series = pd.Series([60.1, 59.9, 60.0, 60.2, 59.8])
    rep = monitor.assess_sensor("S1", short_series)

    assert rep.health_status == "Insufficient Data"
    assert rep.health_score == 0.0
    assert "fewer than 10" in rep.health_explanation.lower()


def test_9_network_health_aggregation():
    """Verify network summary aggregates metrics accurately without skewed denominator."""
    monitor = SensorHealthMonitor()
    mock_reports = {
        "S1": SensorHealthRecord("S1", 90.0, "Healthy", 0.0, 0.1, 0.2, None, "Stable", "", "Normal"),
        "S2": SensorHealthRecord("S2", 70.0, "Warning", 0.2, 0.5, 0.7, 150, "Degrading", "", "Elevated"),
        "S3": SensorHealthRecord("S3", 40.0, "Critical", 0.6, 0.8, 0.95, 145, "Critical Divergence", "", "Immediate Inspection"),
        "S4": SensorHealthRecord("S4", 88.0, "Healthy", 0.0, 0.15, 0.25, None, "Stable", "", "Normal"),
        "S5": SensorHealthRecord("S5", 0.0, "Insufficient Data", 0.0, 0.0, 0.0, None, "Unknown", "", "Verify Telemetry"),
    }

    summary = monitor.summarize_network_health(mock_reports)

    assert isinstance(summary, NetworkHealthSummary)
    assert summary.healthy_count == 2
    assert summary.warning_count == 1
    assert summary.critical_count == 1
    assert summary.insufficient_data_count == 1

    # Overall score should be mean of the 4 valid sensors: (90 + 70 + 40 + 88) / 4 = 72.0
    assert summary.overall_network_health_score == 72.0
    # Most vulnerable sensor should be S3 (score 40.0)
    assert summary.most_vulnerable_sensor == "S3"
    # Highest priority sensor should be S3
    assert summary.highest_priority_sensor == "S3"


def test_10_deterministic_results():
    """Verify identical input series produces identical health score and explanation."""
    np.random.seed(123)
    noise = np.random.normal(0, 0.9, 100)
    readings = pd.Series(60.0 + noise)
    scores = pd.Series(np.clip(noise / 2.0, 0.0, 1.0))

    monitor = SensorHealthMonitor()
    rep1 = monitor.assess_sensor("S1", readings, scores)
    rep2 = monitor.assess_sensor("S1", readings, scores)

    assert rep1.health_score == rep2.health_score
    assert rep1.health_status == rep2.health_status
    assert rep1.anomaly_frequency == rep2.anomaly_frequency
    assert rep1.average_anomaly_score == rep2.average_anomaly_score
    assert rep1.health_explanation == rep2.health_explanation


def test_11_changing_simulation_parameters():
    """Verify health score reflects changes in fault magnitude dynamically."""
    sim = SensorSimulator(seed=42)

    # Run A: Mild drift (+10% of baseline)
    df_mild = sim.generate(n_steps=200, fault={"sensor": "S1", "type": "drift", "start": 100, "magnitude": 6.0})
    # Run B: Severe drift (+50% of baseline)
    df_severe = sim.generate(n_steps=200, fault={"sensor": "S1", "type": "drift", "start": 100, "magnitude": 30.0})

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df_mild)
    scores_mild = detector.score(df_mild)
    scores_severe = detector.score(df_severe)

    monitor = SensorHealthMonitor()
    rep_mild = monitor.assess_sensor("S1", df_mild["S1"], scores_mild["S1"])
    rep_severe = monitor.assess_sensor("S1", df_severe["S1"], scores_severe["S1"])

    # Severe fault must result in strictly lower health score than mild fault
    assert rep_severe.health_score < rep_mild.health_score
    assert rep_severe.average_anomaly_score >= rep_mild.average_anomaly_score


def test_12_invalid_or_empty_input_handling():
    """Verify empty DataFrame or None input is safely handled without raising uncaught exceptions."""
    monitor = SensorHealthMonitor()

    # Empty DataFrame
    empty_df = pd.DataFrame()
    reports_empty = monitor.assess_all(empty_df)
    assert len(reports_empty) == 5
    for rep in reports_empty.values():
        assert rep.health_status == "Insufficient Data"
        assert rep.health_score == 0.0

    # None DataFrame
    reports_none = monitor.assess_all(None)
    assert len(reports_none) == 5
    summary = monitor.summarize_network_health(reports_none)
    assert summary.overall_network_health_score == 0.0
    assert summary.insufficient_data_count == 5
    assert summary.most_vulnerable_sensor is None

    # Single sensor assess with None
    rep_none_single = monitor.assess_sensor("S1", None)
    assert rep_none_single.health_status == "Insufficient Data"
