"""Comprehensive test suite for Phase 4: Executive Reporting & PDF Export Center.

Tests:
1. Normal simulation report (Clean baseline, nominal executive summary).
2. Faulty simulation report (Localized fault, evidence, maintenance actions).
3. Missing telemetry handling.
4. Missing explainability output handling.
5. Missing health information handling.
6. Missing maintenance recommendations handling.
7. PDF generation and validity (valid header %PDF-, non-empty output > 1000 bytes).
8. CSV export compatibility (valid column schema, correct row count).
9. Report generation with no simulation (all None arguments).
10. Deterministic report data apart from generated IDs and timestamps.
11. Verification of all 14 report sections in summary dictionary.
"""

import pytest
import io
import os
import pandas as pd
import numpy as np

from faultmesh.reporting import (
    build_investigation_summary,
    export_summary_csv,
    generate_pdf_report,
    ExecutiveReporter,
    REPORT_DISCLAIMER,
)
from faultmesh.explainer import FaultExplainer
from faultmesh.health import SensorHealthMonitor, SensorHealthRecord
from faultmesh.maintenance import MaintenanceAdvisor, MaintenanceAdvisory


# -----------------------------------------------------------------------------
# Fixtures & Mock Data
# -----------------------------------------------------------------------------
@pytest.fixture
def faulty_pipeline_data():
    """Build mock pipeline outputs for a faulty simulation (S1 drift)."""
    readings_df = pd.DataFrame({
        "time": list(range(100)),
        "S1": [60.0] * 50 + [85.0] * 50,
        "S2": [2.0] * 50 + [2.8] * 50,
        "S3": [5.0] * 100,
        "S4": [30.0] * 100,
        "S5": [10.0] * 100,
    })
    scores_df = pd.DataFrame({
        "time": list(range(100)),
        "S1": [0.1] * 50 + [0.9] * 50,
        "S2": [0.1] * 50 + [0.7] * 50,
        "S3": [0.1] * 100,
        "S4": [0.1] * 100,
        "S5": [0.1] * 100,
    })
    onsets = {"S1": 52, "S2": 55, "S3": None, "S4": None, "S5": None}
    rankings = [
        ("S1", 0.88, {"own_anomaly": 0.9, "detection_earliness": 1.0, "neighbour_support": 0.5, "coverage": 0.5}),
        ("S2", 0.65, {"own_anomaly": 0.7, "detection_earliness": 0.5, "neighbour_support": 0.2, "coverage": 0.4}),
        ("S3", 0.30, {"own_anomaly": 0.1, "detection_earliness": 0.0, "neighbour_support": 0.8, "coverage": 0.8}),
        ("S4", 0.20, {"own_anomaly": 0.1, "detection_earliness": 0.0, "neighbour_support": 0.4, "coverage": 0.4}),
        ("S5", 0.15, {"own_anomaly": 0.1, "detection_earliness": 0.0, "neighbour_support": 0.2, "coverage": 0.4}),
    ]
    cfg_info = {"fault_sensor": "S1", "fault_type": "drift", "fault_start": 50, "fault_magnitude_ratio": 0.5}

    explainer = FaultExplainer()
    inv_rep = explainer.explain(scores_df, onsets, rankings)

    h_mon = SensorHealthMonitor()
    h_reps = h_mon.assess_all(readings_df, scores_df, onsets=onsets)
    net_health = h_mon.summarize_network_health(h_reps)

    m_adv = MaintenanceAdvisor()
    advisories = m_adv.advise_fleet(health_reports=h_reps, suspect_sensor="S1", fault_type="drift", rankings=rankings)
    maint_summary = m_adv.summarize_network_maintenance(advisories)

    benchmark_data = {
        "top1_accuracy": 1.0,
        "top2_accuracy": 1.0,
        "total_scenarios": 75,
        "detection_rate": 1.0,
    }

    return {
        "cfg": cfg_info,
        "readings": readings_df,
        "scores": scores_df,
        "onsets": onsets,
        "rankings": rankings,
        "inv_rep": inv_rep,
        "h_reps": h_reps,
        "net_health": net_health,
        "advisories": advisories,
        "maint_summary": maint_summary,
        "benchmark": benchmark_data,
    }


@pytest.fixture
def normal_pipeline_data():
    """Build mock pipeline outputs for a normal clean simulation."""
    from faultmesh.simulator import SensorSimulator
    from faultmesh.detector import AnomalyDetector, first_onset

    sim = SensorSimulator(seed=42)
    readings_df = sim.generate(n_steps=150, fault=None)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(readings_df)
    scores_df = detector.score(readings_df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    cfg_info = {"fault_sensor": None, "fault_type": None}

    h_mon = SensorHealthMonitor()
    h_reps = h_mon.assess_all(readings_df, scores_df, onsets=onsets)
    net_health = h_mon.summarize_network_health(h_reps)

    m_adv = MaintenanceAdvisor()
    advisories = m_adv.advise_fleet(health_reports=h_reps)
    maint_summary = m_adv.summarize_network_maintenance(advisories)

    return {
        "cfg": cfg_info,
        "readings": readings_df,
        "scores": scores_df,
        "onsets": onsets,
        "rankings": [],
        "inv_rep": None,
        "h_reps": h_reps,
        "net_health": net_health,
        "advisories": advisories,
        "maint_summary": maint_summary,
        "benchmark": None,
    }


# -----------------------------------------------------------------------------
# 1. Normal Simulation Report
# -----------------------------------------------------------------------------
def test_normal_simulation_report(normal_pipeline_data):
    """Verify report compiled from normal simulation produces clean baseline summary."""
    d = normal_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        network_health=d["net_health"],
        maintenance_summary=d["maint_summary"],
        benchmark=d["benchmark"],
    )

    assert "nominal" in summary["executive_summary"].lower()
    assert summary["maintenance_summary"]["overall_attention_level"] == "Nominal"
    for s_id, adv in summary["predictive_maintenance"]["advisories"].items():
        assert adv.priority == "Low"


# -----------------------------------------------------------------------------
# 2. Faulty Simulation Report
# -----------------------------------------------------------------------------
def test_faulty_simulation_report(faulty_pipeline_data):
    """Verify report compiled from faulty simulation localizes root cause to S1."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        network_health=d["net_health"],
        maintenance_summary=d["maint_summary"],
        benchmark=d["benchmark"],
    )

    assert summary["localization"]["primary_suspect"] == "S1"
    assert summary["localization"]["composite_score"] > 0.8
    assert "S1" in summary["executive_summary"]
    assert len(summary["explainable_evidence"]["evidence_items"]) >= 1
    assert summary["benchmark_results"]["available"] is True


# -----------------------------------------------------------------------------
# 3. Missing Telemetry Handling
# -----------------------------------------------------------------------------
def test_missing_telemetry(faulty_pipeline_data):
    """Verify safe fallback when readings_df is None."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=None,
        scores_df=None,
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
    )

    assert summary["simulation_config"]["total_timesteps"] == 0
    assert "preliminary" in summary["executive_summary"].lower() or "standby" in summary["executive_summary"].lower()


# -----------------------------------------------------------------------------
# 4. Missing Explainability Output Handling
# -----------------------------------------------------------------------------
def test_missing_explainability_output(faulty_pipeline_data):
    """Verify report functions when investigation_report is None."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=None,
        health_reports=d["h_reps"],
        advisories=d["advisories"],
    )

    assert summary["localization"]["primary_suspect"] == "S1"
    assert "explainable_evidence" in summary
    assert len(summary["explainable_evidence"]["narrative"]) > 0


# -----------------------------------------------------------------------------
# 5. Missing Health Information Handling
# -----------------------------------------------------------------------------
def test_missing_health_information(faulty_pipeline_data):
    """Verify report functions when health_reports is None."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=None,
        advisories=d["advisories"],
    )

    assert summary["sensor_health"]["reports"] == {}
    assert summary["sensor_health"]["network_health_score"] >= 0.0


# -----------------------------------------------------------------------------
# 6. Missing Maintenance Recommendations Handling
# -----------------------------------------------------------------------------
def test_missing_maintenance_recommendations(faulty_pipeline_data):
    """Verify report functions when advisories is None."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=None,
    )

    assert summary["predictive_maintenance"]["advisories"] == {}
    assert "maintenance_summary" in summary


# -----------------------------------------------------------------------------
# 7. PDF Generation and File Validity
# -----------------------------------------------------------------------------
def test_pdf_generation_valid_and_non_empty(faulty_pipeline_data):
    """Verify PDF report generation produces a valid, non-empty PDF file."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        network_health=d["net_health"],
        maintenance_summary=d["maint_summary"],
        benchmark=d["benchmark"],
    )

    pdf_buf = io.BytesIO()
    generate_pdf_report(summary, pdf_buf)
    pdf_bytes = pdf_buf.getvalue()

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


# -----------------------------------------------------------------------------
# 8. CSV Export Compatibility
# -----------------------------------------------------------------------------
def test_csv_export_compatibility(faulty_pipeline_data):
    """Verify CSV export produces valid schema and correct row count."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
    )

    buf = io.StringIO()
    export_summary_csv(summary, buf)
    buf.seek(0)
    df = pd.read_csv(buf)

    assert len(df) == 5
    expected_cols = [
        "sensor", "subsystem", "suspect_rank", "composite_score",
        "alarm_onset", "peak_anomaly_score", "mean_reading", "latest_reading",
        "health_score", "health_status", "maintenance_priority", "recommended_action",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in exported CSV"


# -----------------------------------------------------------------------------
# 9. Report Generation With No Simulation
# -----------------------------------------------------------------------------
def test_report_generation_with_no_simulation():
    """Verify build_investigation_summary with all None arguments returns a valid report."""
    summary = build_investigation_summary()

    assert isinstance(summary, dict)
    assert "report_id" in summary
    assert "executive_summary" in summary
    assert "limitations_and_disclaimer" in summary
    assert len(summary["telemetry_stats"]) == 5

    # Should also generate valid PDF without error
    pdf_buf = io.BytesIO()
    generate_pdf_report(summary, pdf_buf)
    assert len(pdf_buf.getvalue()) > 500


# -----------------------------------------------------------------------------
# 10. Deterministic Report Data
# -----------------------------------------------------------------------------
def test_deterministic_report_data(faulty_pipeline_data):
    """Verify identical inputs yield identical report payloads apart from ID/timestamp."""
    d = faulty_pipeline_data
    s1 = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        report_id="TEST-RPT-001",
    )
    s2 = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        report_id="TEST-RPT-001",
    )

    assert s1["localization"]["composite_score"] == s2["localization"]["composite_score"]
    assert s1["localization"]["primary_suspect"] == s2["localization"]["primary_suspect"]
    assert s1["telemetry_stats"] == s2["telemetry_stats"]
    assert s1["report_id"] == s2["report_id"]


# -----------------------------------------------------------------------------
# 11. All 14 Specified Report Sections Present
# -----------------------------------------------------------------------------
def test_all_14_report_sections_present(faulty_pipeline_data):
    """Verify all 14 mandatory sections exist in the summary dictionary."""
    d = faulty_pipeline_data
    summary = build_investigation_summary(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
        network_health=d["net_health"],
        maintenance_summary=d["maint_summary"],
        benchmark=d["benchmark"],
    )

    required_sections = [
        "report_id",                     # 1. Report ID & timestamp
        "generated_at",
        "project_name",                  # 2. Project name & title
        "report_title",
        "executive_summary",             # 3. Executive summary
        "network_overview",              # 4. Machine & network overview
        "simulation_config",             # 5. Simulation & fault config
        "anomaly_detection",             # 6. Anomaly detection findings
        "localization",                  # 7. Root-cause localization rankings
        "explainable_evidence",          # 8. Explainable investigation evidence
        "sensor_health",                 # 9. Sensor health assessment
        "predictive_maintenance",        # 10. Predictive maintenance priorities
        "technician_actions",            # 11. Recommended technician follow-up actions
        "maintenance_summary",           # 12. Network-level maintenance summary
        "benchmark_results",             # 13. Benchmark results
        "limitations_and_disclaimer",    # 14. Limitations & diagnostic disclaimer
    ]

    for sec in required_sections:
        assert sec in summary, f"Mandatory section '{sec}' is missing from report dictionary"


# -----------------------------------------------------------------------------
# 12. ExecutiveReporter Class Wrapper
# -----------------------------------------------------------------------------
def test_executive_reporter_class(faulty_pipeline_data):
    """Verify ExecutiveReporter coordinator compiles and exports PDF and CSV."""
    d = faulty_pipeline_data
    reporter = ExecutiveReporter()
    rep_dict = reporter.compile_report(
        cfg_info=d["cfg"],
        readings_df=d["readings"],
        scores_df=d["scores"],
        onsets=d["onsets"],
        rankings=d["rankings"],
        investigation_report=d["inv_rep"],
        health_reports=d["h_reps"],
        advisories=d["advisories"],
    )

    csv_data = reporter.export_csv(rep_dict)
    assert isinstance(csv_data, str)
    assert "S1" in csv_data

    pdf_bytes = reporter.export_pdf(rep_dict)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
