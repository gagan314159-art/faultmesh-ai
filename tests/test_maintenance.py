"""Comprehensive test suite for Phase 3: Predictive Maintenance Recommendation Engine.

Tests:
1. All five sensors (S1 to S5) evaluated.
2. Healthy sensor recommendations (Low priority, routine review).
3. Warning sensor recommendations (High / Medium priority, increased monitoring).
4. Critical sensor recommendations (Critical priority, immediate review).
5. Drift fault recommendations (Sensor calibration review).
6. Spike fault recommendations (Connection & signal-quality inspection).
7. Stuck fault recommendations (Sensor replacement assessment).
8. Insufficient data handling (Insufficient Data priority & data collection).
9. Missing health information handling.
10. Missing localization evidence handling.
11. Network-level maintenance summary calculation.
12. Maintenance priority ranking order.
13. Deterministic results for identical inputs.
14. Changing simulation parameters alters recommendations.
15. Invalid or empty input handling.
"""

import pytest
import pandas as pd
import numpy as np

from faultmesh.maintenance import (
    MaintenanceAdvisor,
    MaintenanceAdvisory,
    MaintenanceAction,
    NetworkMaintenanceSummary,
    MAINTENANCE_DISCLAIMER,
)
from faultmesh.health import SensorHealthRecord, SensorHealthMonitor
from faultmesh.simulator import SensorSimulator
from dashboard.utils import SimConfig, run_simulation
from faultmesh.processing import FeatureProcessor
from faultmesh.detector import AnomalyDetector
from faultmesh.localizer import FaultLocalizer
from faultmesh.graph import SensorNetworkGraph
from faultmesh.explainer import FaultExplainer


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------
@pytest.fixture
def advisor() -> MaintenanceAdvisor:
    return MaintenanceAdvisor()


def make_health_record(
    sensor_id: str,
    health_score: float = 95.0,
    health_status: str = "Healthy",
    anomaly_frequency: float = 0.0,
    max_anomaly: float = 0.1,
    avg_anomaly: float = 0.05,
    onset: int = None,
    trend: str = "Stable",
) -> SensorHealthRecord:
    return SensorHealthRecord(
        sensor_id=sensor_id,
        health_score=health_score,
        health_status=health_status,
        anomaly_frequency=anomaly_frequency,
        average_anomaly_score=avg_anomaly,
        maximum_anomaly_score=max_anomaly,
        detection_onset=onset,
        recent_trend=trend,
        health_explanation=f"Test record for {sensor_id}",
        recommended_monitoring_priority="Normal" if health_status == "Healthy" else "Immediate",
        current_value=10.0,
        baseline_value=10.0,
        unit="units",
        sensor_name=f"Sensor {sensor_id}",
    )


# -----------------------------------------------------------------------------
# 1. Test All Five Sensors
# -----------------------------------------------------------------------------
def test_1_all_five_sensors(advisor: MaintenanceAdvisor):
    """Verify that all five sensors (S1-S5) receive maintenance advisories."""
    reports = {s: make_health_record(s) for s in ["S1", "S2", "S3", "S4", "S5"]}
    fleet = advisor.advise_fleet(health_reports=reports)

    assert len(fleet) == 5
    for s_id in ["S1", "S2", "S3", "S4", "S5"]:
        assert s_id in fleet
        adv = fleet[s_id]
        assert isinstance(adv, MaintenanceAdvisory)
        assert adv.sensor_id == s_id
        assert isinstance(adv.health_score, float)
        assert adv.priority in ("Critical", "High", "Medium", "Low", "Insufficient Data")
        assert len(adv.recommended_action) > 0
        assert len(adv.reason) > 0
        assert len(adv.supporting_evidence) >= 1
        assert len(adv.confidence_category) > 0
        assert len(adv.limitations) >= 1
        assert len(adv.actions) >= 1
        assert len(adv.inspection_checklist) >= 1
        assert adv.safety_disclaimer == MAINTENANCE_DISCLAIMER


# -----------------------------------------------------------------------------
# 2. Healthy Sensor Recommendations
# -----------------------------------------------------------------------------
def test_2_healthy_sensor_recommendations(advisor: MaintenanceAdvisor):
    """Verify that healthy sensors receive Low priority and routine maintenance."""
    healthy_rep = make_health_record("S1", health_score=96.0, health_status="Healthy", anomaly_frequency=0.0)
    adv = advisor.recommend("S1", health_report=healthy_rep)

    assert adv.priority == "Low"
    assert "routine" in adv.recommended_action.lower()
    assert adv.urgency_score <= 20.0
    assert "30 days" in adv.suggested_follow_up or "routine" in adv.suggested_follow_up.lower()
    assert "nominal" in adv.reason.lower() or "healthy" in adv.reason.lower()


# -----------------------------------------------------------------------------
# 3. Warning Sensor Recommendations
# -----------------------------------------------------------------------------
def test_3_warning_sensor_recommendations(advisor: MaintenanceAdvisor):
    """Verify that warning sensors receive High or Medium priority and increased monitoring/inspection."""
    warn_rep = make_health_record(
        "S2", health_score=72.0, health_status="Warning", anomaly_frequency=0.18, max_anomaly=0.68, trend="Degrading"
    )
    adv = advisor.recommend("S2", health_report=warn_rep, fault_type_hint="drift")

    assert adv.priority in ("High", "Medium")
    assert adv.urgency_score >= 25.0
    assert any(w in adv.recommended_action.lower() for w in ["calibration", "monitoring", "inspection"])


# -----------------------------------------------------------------------------
# 4. Critical Sensor Recommendations
# -----------------------------------------------------------------------------
def test_4_critical_sensor_recommendations(advisor: MaintenanceAdvisor):
    """Verify that critical sensors receive Critical priority and immediate follow-up."""
    crit_rep = make_health_record(
        "S3", health_score=35.0, health_status="Critical", anomaly_frequency=0.65, max_anomaly=0.95, onset=155
    )
    adv = advisor.recommend("S3", health_report=crit_rep, fault_type_hint="drift")

    assert adv.priority == "Critical"
    assert adv.urgency_score >= 65.0
    assert "immediate" in adv.suggested_follow_up.lower()
    assert "critical" in adv.reason.lower()


# -----------------------------------------------------------------------------
# 5. Drift Fault Recommendations
# -----------------------------------------------------------------------------
def test_5_drift_fault_recommendations(advisor: MaintenanceAdvisor):
    """Verify that drift faults produce calibration review recommendations."""
    drift_rep = make_health_record(
        "S3", health_score=45.0, health_status="Critical", anomaly_frequency=0.50, max_anomaly=0.85
    )
    adv = advisor.recommend("S3", health_report=drift_rep, fault_type_hint="drift")

    assert "calibration" in adv.recommended_action.lower()
    assert len(adv.actions) >= 1
    action_titles = [a.title.lower() for a in adv.actions]
    assert any("calibration" in t or "regulator" in t or "manifold" in t for t in action_titles)


# -----------------------------------------------------------------------------
# 6. Spike Fault Recommendations
# -----------------------------------------------------------------------------
def test_6_spike_fault_recommendations(advisor: MaintenanceAdvisor):
    """Verify that spike faults produce connection and signal-quality inspection recommendations."""
    spike_rep = make_health_record(
        "S1", health_score=55.0, health_status="Warning", anomaly_frequency=0.20, max_anomaly=0.92, avg_anomaly=0.15
    )
    adv = advisor.recommend("S1", health_report=spike_rep, fault_type_hint="spike")

    assert "signal" in adv.recommended_action.lower() or "connection" in adv.recommended_action.lower()
    assert len(adv.actions) >= 1
    action_titles = [a.title.lower() for a in adv.actions]
    assert any("shield" in t or "continuity" in t or "noise" in t or "grounding" in t for t in action_titles)


# -----------------------------------------------------------------------------
# 7. Stuck Fault Recommendations
# -----------------------------------------------------------------------------
def test_7_stuck_fault_recommendations(advisor: MaintenanceAdvisor):
    """Verify that stuck faults produce sensor replacement assessment recommendations."""
    stuck_rep = make_health_record(
        "S4", health_score=40.0, health_status="Critical", anomaly_frequency=0.60, max_anomaly=0.88
    )
    adv = advisor.recommend("S4", health_report=stuck_rep, fault_type_hint="stuck")

    assert "replacement" in adv.recommended_action.lower() or "responsiveness" in adv.recommended_action.lower()
    assert len(adv.actions) >= 1
    action_titles = [a.title.lower() for a in adv.actions]
    assert any("replacement" in t or "transducer" in t or "freeze" in t or "lockup" in t for t in action_titles)


# -----------------------------------------------------------------------------
# 8. Insufficient Data Handling
# -----------------------------------------------------------------------------
def test_8_insufficient_data_handling(advisor: MaintenanceAdvisor):
    """Verify that sensors with insufficient data produce Insufficient Data priority."""
    insuf_rep = SensorHealthRecord(sensor_id="S5", health_status="Insufficient Data", health_score=0.0)
    adv = advisor.recommend("S5", health_report=insuf_rep)

    assert adv.priority == "Insufficient Data"
    assert "data collection" in adv.recommended_action.lower()
    assert "insufficient" in adv.reason.lower()
    assert adv.confidence_category.lower() in ("indeterminate", "low")


# -----------------------------------------------------------------------------
# 9. Missing Health Information
# -----------------------------------------------------------------------------
def test_9_missing_health_information(advisor: MaintenanceAdvisor):
    """Verify safe fallback when health report is None."""
    adv = advisor.recommend("S2", health_report=None)

    assert isinstance(adv, MaintenanceAdvisory)
    assert adv.priority == "Insufficient Data"
    assert "data collection" in adv.recommended_action.lower()
    assert len(adv.actions) >= 1


# -----------------------------------------------------------------------------
# 10. Missing Localization Evidence
# -----------------------------------------------------------------------------
def test_10_missing_localization_evidence(advisor: MaintenanceAdvisor):
    """Verify that recommendations function properly when localization rankings are absent."""
    rep = make_health_record("S1", health_score=60.0, health_status="Warning", anomaly_frequency=0.25)
    adv = advisor.recommend("S1", health_report=rep, rankings=None, investigation_report=None)

    assert isinstance(adv, MaintenanceAdvisory)
    assert adv.priority in ("Critical", "High")
    assert adv.health_metrics["health_score"] == 60.0
    assert adv.health_metrics.get("localization_rank") is None


# -----------------------------------------------------------------------------
# 11. Network-Level Maintenance Summary
# -----------------------------------------------------------------------------
def test_11_network_level_maintenance_summary(advisor: MaintenanceAdvisor):
    """Verify calculation of network-level maintenance counts and attention level."""
    reports = {
        "S1": make_health_record("S1", health_score=40.0, health_status="Critical"),
        "S2": make_health_record("S2", health_score=75.0, health_status="Warning"),
        "S3": make_health_record("S3", health_score=85.0, health_status="Healthy"),
        "S4": make_health_record("S4", health_score=95.0, health_status="Healthy"),
        "S5": SensorHealthRecord(sensor_id="S5", health_status="Insufficient Data", health_score=0.0),
    }

    fleet = advisor.advise_fleet(health_reports=reports)
    summary = advisor.summarize_network_maintenance(fleet)

    assert isinstance(summary, NetworkMaintenanceSummary)
    assert summary.total_sensors_evaluated == 5
    assert summary.critical_cases == 1
    assert summary.high_priority_cases >= 1
    assert summary.low_priority_cases >= 1
    assert summary.insufficient_data_cases == 1
    assert "S5" in summary.sensors_requiring_more_data
    assert summary.highest_priority_sensor == "S1"
    assert summary.overall_maintenance_attention_level == "Critical"
    assert len(summary.ranked_action_list) == 5


# -----------------------------------------------------------------------------
# 12. Maintenance Priority Ranking Order
# -----------------------------------------------------------------------------
def test_12_maintenance_priority_ranking(advisor: MaintenanceAdvisor):
    """Verify that ranked_action_list is ordered with Critical before High/Medium/Low."""
    reports = {
        "S1": make_health_record("S1", health_score=95.0, health_status="Healthy"),
        "S2": make_health_record("S2", health_score=35.0, health_status="Critical"),
        "S3": make_health_record("S3", health_score=72.0, health_status="Warning"),
    }

    fleet = advisor.advise_fleet(health_reports=reports)
    summary = advisor.summarize_network_maintenance(fleet)

    ranked_sensors = [item["sensor_id"] for item in summary.ranked_action_list]
    # S2 (Critical) must be ranked ahead of S3 (Warning) and S1 (Healthy)
    assert ranked_sensors.index("S2") < ranked_sensors.index("S3")
    assert ranked_sensors.index("S3") < ranked_sensors.index("S1")


# -----------------------------------------------------------------------------
# 13. Deterministic Results
# -----------------------------------------------------------------------------
def test_13_deterministic_results(advisor: MaintenanceAdvisor):
    """Verify identical inputs yield identical maintenance advisories."""
    rep = make_health_record("S3", health_score=50.0, health_status="Critical", anomaly_frequency=0.45)

    adv1 = advisor.recommend("S3", health_report=rep, fault_type_hint="drift")
    adv2 = advisor.recommend("S3", health_report=rep, fault_type_hint="drift")

    assert adv1.priority == adv2.priority
    assert adv1.recommended_action == adv2.recommended_action
    assert adv1.urgency_score == adv2.urgency_score
    assert adv1.reason == adv2.reason
    assert adv1.suggested_follow_up == adv2.suggested_follow_up


# -----------------------------------------------------------------------------
# 14. Changing Simulation Parameters
# -----------------------------------------------------------------------------
def test_14_changing_simulation_parameters(advisor: MaintenanceAdvisor):
    """Verify that changing simulation fault targets produces corresponding advisories."""
    # Simulation A: Fault in S1 (drift)
    cfg_a = SimConfig(total_steps=200, fault_sensor="S1", fault_type="drift", fault_magnitude_ratio=0.5, fault_start=120)
    df_a, _ = run_simulation(cfg_a)
    mon = SensorHealthMonitor()
    health_a = mon.assess_all(df_a)
    fleet_a = advisor.advise_fleet(health_reports=health_a, suspect_sensor="S1", fault_type="drift")

    # Simulation B: Fault in S3 (spike)
    cfg_b = SimConfig(total_steps=200, fault_sensor="S3", fault_type="spike", fault_magnitude_ratio=0.5, fault_start=120, spike_duration=10)
    df_b, _ = run_simulation(cfg_b)
    health_b = mon.assess_all(df_b)
    fleet_b = advisor.advise_fleet(health_reports=health_b, suspect_sensor="S3", fault_type="spike")

    # Fleet A should prioritize S1; Fleet B should prioritize S3
    summary_a = advisor.summarize_network_maintenance(fleet_a)
    summary_b = advisor.summarize_network_maintenance(fleet_b)

    assert summary_a.highest_priority_sensor == "S1"
    assert summary_b.highest_priority_sensor == "S3"
    assert "calibration" in fleet_a["S1"].recommended_action.lower()
    assert "signal" in fleet_b["S3"].recommended_action.lower() or "connection" in fleet_b["S3"].recommended_action.lower()


# -----------------------------------------------------------------------------
# 15. Invalid or Empty Input Handling
# -----------------------------------------------------------------------------
def test_15_invalid_or_empty_input_handling(advisor: MaintenanceAdvisor):
    """Verify safe handling of empty dicts, unknown sensor IDs, and empty inputs."""
    # Empty fleet
    empty_fleet = advisor.advise_fleet(health_reports={})
    assert len(empty_fleet) == 5  # Evaluates default S1-S5 safely
    for adv in empty_fleet.values():
        assert adv.priority == "Insufficient Data"

    summary = advisor.summarize_network_maintenance(empty_fleet)
    assert summary.total_sensors_evaluated == 5
    assert summary.insufficient_data_cases == 5
    assert summary.overall_maintenance_attention_level == "Insufficient Data"

    # Unknown sensor name
    unknown_adv = advisor.recommend("UNKNOWN_SENSOR", health_report=None)
    assert unknown_adv.priority == "Insufficient Data"


# -----------------------------------------------------------------------------
# Backward Compatibility Tests
# -----------------------------------------------------------------------------
def test_maintenance_advisor_critical_priority(advisor: MaintenanceAdvisor):
    """Backward compatibility test for critical health priority."""
    mock_health = SensorHealthRecord(
        sensor_id="S1",
        health_score=25.0,
        health_status="Critical",
        anomaly_frequency=0.55,
        maximum_anomaly_score=0.90,
    )
    advisory = advisor.recommend(sensor="S1", health_report=mock_health, fault_type_hint="drift")
    assert advisory.priority == "Critical"
    assert advisory.urgency_score >= 75.0
    assert len(advisory.actions) >= 1


def test_maintenance_advisor_healthy_priority(advisor: MaintenanceAdvisor):
    """Backward compatibility test for healthy priority."""
    mock_health = SensorHealthRecord(
        sensor_id="S2",
        health_score=95.0,
        health_status="Healthy",
        anomaly_frequency=0.0,
        maximum_anomaly_score=0.1,
    )
    advisory = advisor.recommend(sensor="S2", health_report=mock_health)
    assert advisory.priority == "Low"
    assert advisory.urgency_score <= 20.0
