"""Comprehensive tests for Explainable Fault Investigation Engine (faultmesh/explainer.py).

Verifies:
- Correct suspected sensor identification.
- Different sensor rankings.
- Detection onset comparisons.
- Neighboring sensor evidence.
- Missing anomaly scores.
- Missing detection onsets.
- Normal scenarios with no detected faults.
- Insufficient evidence handling.
- Dynamic explanations for all five sensors (S1, S2, S3, S4, S5).
"""

import pytest
import pandas as pd
import numpy as np

from faultmesh.explainer import FaultExplainer, InvestigationReport
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer


def test_explainer_initialization():
    """Verify FaultExplainer initializes with default and custom graphs."""
    explainer = FaultExplainer()
    assert explainer.graph is not None
    assert len(explainer.graph.nodes) == 5

    custom_graph = SensorNetworkGraph(nodes=["A", "B"], edges=[("A", "B")])
    custom_explainer = FaultExplainer(network_graph=custom_graph)
    assert len(custom_explainer.graph.nodes) == 2


def test_suspected_sensor_identification():
    """Verify suspect sensor and ranking are correctly extracted from rankings."""
    localizer = FaultLocalizer()
    anomaly_scores = {"S3": 0.85, "S1": 0.40, "S2": 0.35, "S4": 0.30, "S5": 0.25}
    onsets = {"S3": 151, "S1": 154, "S2": 155, "S4": None, "S5": None}
    rankings = localizer.rank_suspects(anomaly_scores, onsets)

    explainer = FaultExplainer()
    report = explainer.explain(onsets=onsets, rankings=rankings, anomaly_scores=anomaly_scores)

    assert isinstance(report, InvestigationReport)
    assert report.suspected_sensor == "S3"
    assert report.localization_rank == 1
    assert report.localization_score == round(rankings[0][1], 4)
    assert report.anomaly_score == 0.85
    assert report.detection_onset == 151
    assert "S3" in report.investigation_summary
    assert report.confidence_category in ("Strong", "Moderate")


def test_different_sensor_rankings():
    """Verify explainer identifies different sensors when they rank highest."""
    explainer = FaultExplainer()

    # Case 1: S2 ranks #1
    rankings_s2 = [
        ("S2", 0.82, {"own_anomaly": 0.85, "detection_earliness": 0.9, "neighbour_support": 0.5, "coverage": 0.5}),
        ("S1", 0.60, {"own_anomaly": 0.50, "detection_earliness": 0.4, "neighbour_support": 0.3, "coverage": 0.4}),
    ]
    onsets_s2 = {"S2": 150, "S1": 153, "S3": 154, "S4": None, "S5": None}
    report_s2 = explainer.explain(onsets=onsets_s2, rankings=rankings_s2)
    assert report_s2.suspected_sensor == "S2"
    assert report_s2.localization_rank == 1

    # Case 2: S5 ranks #1
    rankings_s5 = [
        ("S5", 0.79, {"own_anomaly": 0.80, "detection_earliness": 0.9, "neighbour_support": 0.4, "coverage": 0.6}),
        ("S4", 0.55, {"own_anomaly": 0.45, "detection_earliness": 0.3, "neighbour_support": 0.2, "coverage": 0.4}),
    ]
    onsets_s5 = {"S5": 148, "S4": 152, "S3": 153, "S1": None, "S2": None}
    report_s5 = explainer.explain(onsets=onsets_s5, rankings=rankings_s5)
    assert report_s5.suspected_sensor == "S5"
    assert report_s5.localization_rank == 1


def test_detection_onset_comparisons():
    """Verify timing comparisons correctly calculate delays relative to suspect."""
    explainer = FaultExplainer()
    # S1 alarmed at 150, S2 at 153 (+3 delay), S3 at 148 (-2 lead, premature)
    onsets = {"S1": 150, "S2": 153, "S3": 148, "S4": None, "S5": None}
    rankings = [
        ("S1", 0.75, {"own_anomaly": 0.8, "detection_earliness": 0.8, "neighbour_support": 0.4, "coverage": 0.4}),
        ("S3", 0.70, {"own_anomaly": 0.7, "detection_earliness": 1.0, "neighbour_support": 0.3, "coverage": 0.7}),
    ]

    report = explainer.explain(onsets=onsets, rankings=rankings)
    timing_map = {item["sensor"]: item for item in report.timing_comparisons}

    assert timing_map["S1"]["onset"] == 150
    assert timing_map["S1"]["delay_relative_to_suspect"] == "0 (Reference)"

    assert timing_map["S2"]["onset"] == 153
    assert "+3" in timing_map["S2"]["delay_relative_to_suspect"]
    assert "delayed cascade" in timing_map["S2"]["timing_status"].lower()

    assert timing_map["S3"]["onset"] == 148
    assert "-2" in timing_map["S3"]["delay_relative_to_suspect"]
    assert "preceded" in timing_map["S3"]["timing_status"].lower()

    assert timing_map["S4"]["onset"] is None
    assert "nominal" in timing_map["S4"]["timing_status"].lower()


def test_neighboring_sensor_evidence():
    """Verify connected neighbors in the topology are evaluated for causal propagation."""
    explainer = FaultExplainer()
    # In standard topology, S3 connects to S1, S2, S4, S5
    onsets = {"S3": 150, "S1": 152, "S2": 154, "S4": None, "S5": None}
    rankings = [
        ("S3", 0.86, {"own_anomaly": 0.9, "detection_earliness": 1.0, "neighbour_support": 0.6, "coverage": 0.8}),
    ]

    report = explainer.explain(onsets=onsets, rankings=rankings)

    # S1 and S2 are connected neighbors that alarmed after S3 -> supporting sensors
    assert "S1" in report.supporting_sensors
    assert "S2" in report.supporting_sensors
    assert len(report.supporting_sensors) == 2

    # Check neighbor evidence structure
    n_map = {item["neighbor"]: item for item in report.neighbor_evidence}
    assert n_map["S1"]["supports_suspect"] is True
    assert "downstream" in n_map["S1"]["relationship"].lower()
    assert n_map["S4"]["supports_suspect"] is False
    assert "unaffected" in n_map["S4"]["relationship"].lower()


def test_missing_anomaly_scores():
    """Verify explainer handles None or missing anomaly scores gracefully."""
    explainer = FaultExplainer()
    rankings = [
        ("S1", 0.70, {"own_anomaly": 0.75, "detection_earliness": 0.8, "neighbour_support": 0.4, "coverage": 0.4}),
    ]
    onsets = {"S1": 150, "S2": 152, "S3": None, "S4": None, "S5": None}

    # Pass None for scores_df and anomaly_scores
    report = explainer.explain(scores_df=None, onsets=onsets, rankings=rankings, anomaly_scores=None)

    assert report.suspected_sensor == "S1"
    # Should fall back to breakdown own_anomaly
    assert report.anomaly_score == 0.75
    assert report.confidence_category in ("Strong", "Moderate")


def test_missing_detection_onsets():
    """Verify explainer handles missing or all-None onsets gracefully."""
    explainer = FaultExplainer()
    rankings = [
        ("S1", 0.65, {"own_anomaly": 0.70, "detection_earliness": 0.0, "neighbour_support": 0.0, "coverage": 0.4}),
    ]
    # onsets dict has all None
    onsets = {"S1": None, "S2": None, "S3": None, "S4": None, "S5": None}

    report = explainer.explain(onsets=onsets, rankings=rankings)

    assert report.suspected_sensor == "S1"
    assert report.detection_onset is None
    assert report.confidence_category == "Insufficient"
    assert report.supporting_sensors == []


def test_normal_scenarios_no_detected_faults():
    """Verify explainer gracefully handles clean normal scenarios without errors."""
    explainer = FaultExplainer()
    scores_df = pd.DataFrame({
        "time": list(range(200)),
        "S1": [0.12] * 200,
        "S2": [0.15] * 200,
        "S3": [0.10] * 200,
        "S4": [0.08] * 200,
        "S5": [0.11] * 200,
    })
    onsets = {"S1": None, "S2": None, "S3": None, "S4": None, "S5": None}
    rankings = [
        ("S1", 0.20, {"own_anomaly": 0.12, "detection_earliness": 0.0, "neighbour_support": 0.0, "coverage": 0.4}),
    ]

    report = explainer.explain(scores_df=scores_df, onsets=onsets, rankings=rankings)

    assert report.suspected_sensor is None
    assert report.confidence_category == "Insufficient"
    assert "normal" in report.investigation_summary.lower()
    assert len(report.evidence) >= 2


def test_insufficient_evidence_handling():
    """Verify empty rankings or completely empty inputs produce Insufficient category."""
    explainer = FaultExplainer()

    # Empty rankings list
    report1 = explainer.explain(rankings=[])
    assert report1.suspected_sensor is None
    assert report1.confidence_category == "Insufficient"

    # None inputs
    report2 = explainer.explain(scores_df=None, onsets=None, rankings=None, anomaly_scores=None)
    assert report2.suspected_sensor is None
    assert report2.confidence_category == "Insufficient"


def test_dynamic_explanations_for_all_five_sensors():
    """Verify explainer dynamically generates valid diagnostic reports for all 5 sensors."""
    explainer = FaultExplainer()
    sensors = ["S1", "S2", "S3", "S4", "S5"]

    for s in sensors:
        # Construct scenario where sensor 's' is injected and alarms first
        other_sensors = [x for x in sensors if x != s]
        onsets = {s: 150}
        # Alarm one connected neighbor with delay
        nbs = explainer.graph.get_neighbors(s)
        if nbs:
            onsets[nbs[0]] = 154
        for o in other_sensors:
            if o not in onsets:
                onsets[o] = None

        rankings = [
            (s, 0.85, {"own_anomaly": 0.88, "detection_earliness": 1.0, "neighbour_support": 0.5, "coverage": 0.5}),
            (other_sensors[0], 0.50, {"own_anomaly": 0.40, "detection_earliness": 0.2, "neighbour_support": 0.2, "coverage": 0.4}),
        ]
        anomaly_scores = {s: 0.88, other_sensors[0]: 0.40}

        report = explainer.explain(onsets=onsets, rankings=rankings, anomaly_scores=anomaly_scores)

        assert report.suspected_sensor == s, f"Failed for sensor {s}"
        assert report.localization_rank == 1
        assert report.detection_onset == 150
        assert report.confidence_category in ("Strong", "Moderate")
        assert s in report.investigation_summary
        assert len(report.evidence) >= 3
        assert len(report.limitations) >= 3
        assert report.localization_score == 0.85
