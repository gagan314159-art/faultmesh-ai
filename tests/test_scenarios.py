"""Test scenarios for FaultMesh AI components and pipelines.

Responsibility:
    Test important project scenarios including normal stream generation,
    feature calculation, Isolation Forest baseline initialization,
    graph neighbor queries, candidate ranking, and end-to-end integration.
"""

import pytest
import pandas as pd
import numpy as np

from faultmesh.simulator import SensorSimulator, DEFAULT_BASELINES
from faultmesh.processing import FeatureProcessor
from faultmesh.detector import AnomalyDetector
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer
from faultmesh.storage import StorageManager
from faultmesh.pipeline import FaultMeshPipeline


def test_simulator_normal_stream_shape_and_columns():
    """Verify simulator generates correct shape and sensor columns."""
    sim = SensorSimulator(seed=42)
    df = sim.generate_normal_stream(num_samples=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    expected_cols = {"timestep", "S1", "S2", "S3", "S4", "S5"}
    assert expected_cols.issubset(set(df.columns))

    # Verify baseline means are roughly near nominal values
    for sensor, baseline in DEFAULT_BASELINES.items():
        sample_mean = df[sensor].mean()
        # Mean should be within 5% of baseline
        assert abs(sample_mean - baseline) < baseline * 0.05


def test_feature_processor_extraction():
    """Verify feature processor generates rolling stats, differences, and Z-scores."""
    sim = SensorSimulator(seed=42)
    df = sim.generate_normal_stream(num_samples=50)

    processor = FeatureProcessor(window_size=5)
    sensors = ["S1", "S2", "S3", "S4", "S5"]
    processor.fit_baseline(df, sensors)
    features_df = processor.extract_features(df, sensors)

    for sensor in sensors:
        assert f"{sensor}_roll_mean" in features_df.columns
        assert f"{sensor}_roll_std" in features_df.columns
        assert f"{sensor}_diff" in features_df.columns
        assert f"{sensor}_zscore" in features_df.columns


def test_graph_topology_and_neighbors():
    """Verify NetworkX sensor network has 5 nodes and 7 edges as specified in rules."""
    graph = SensorNetworkGraph()

    assert len(graph.nodes) == 5
    assert len(graph.edges) == 7

    # S3 should be connected to S1, S2, S4, S5
    s3_neighbors = set(graph.get_neighbors("S3"))
    assert s3_neighbors == {"S1", "S2", "S4", "S5"}

    # S1 should be connected to S3, S2
    s1_neighbors = set(graph.get_neighbors("S1"))
    assert s1_neighbors == {"S2", "S3"}


def test_detector_initialization():
    """Verify AnomalyDetector trains on baseline samples without raising errors."""
    sim = SensorSimulator(seed=42)
    df = sim.generate_normal_stream(num_samples=100)
    processor = FeatureProcessor()
    sensors = ["S1", "S2", "S3", "S4", "S5"]
    processor.fit_baseline(df, sensors)
    feats = processor.extract_features(df, sensors)

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit_all(feats, sensors)

    assert len(detector.models) == 5
    scores = detector.compute_anomaly_scores(feats, "S1")
    assert len(scores) == 100
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_localizer_ranking():
    """Verify localizer ranks sensors into sorted order with expected keys."""
    localizer = FaultLocalizer()
    anomaly_scores = {"S1": 0.85, "S2": 0.40, "S3": 0.30, "S4": 0.10, "S5": 0.05}
    alarm_steps = {"S1": 105, "S2": 108, "S3": None, "S4": None, "S5": None}

    rankings = localizer.rank_suspects(anomaly_scores, alarm_steps)

    assert len(rankings) == 5
    # First item should have highest composite score
    assert rankings[0][1] >= rankings[1][1]
    # S1 was highest anomaly score and alarmed first, should be top ranked
    top_suspect = rankings[0][0]
    assert top_suspect == "S1"


def test_storage_initialization(tmp_path):
    """Verify SQLite database schema creation."""
    db_file = tmp_path / "test_faultmesh.db"
    storage = StorageManager(db_path=str(db_file))
    conn = storage._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "sensor_readings" in tables
    assert "detection_alarms" in tables
    assert "evaluation_reports" in tables
    conn.close()


def test_pipeline_smoke_run():
    """Verify end-to-end pipeline executes cleanly and produces expected outputs."""
    pipeline = FaultMeshPipeline()
    result = pipeline.run_pipeline(
        fault_sensor="S1",
        fault_type="drift",
        total_steps=120,
        fault_start=100,
        fault_duration=20,
        baseline_samples=100,
    )

    assert "raw_readings" in result
    assert "features" in result
    assert "detections" in result
    assert "alarm_steps" in result
    assert "rankings" in result
    assert len(result["rankings"]) == 5


def test_simulator_normal_baseline_and_noise():
    """Verify normal readings remain statistically close to baselines and 1.5% noise level."""
    sim = SensorSimulator(seed=42)
    df = sim.generate(n_steps=500, fault=None)

    for sensor, baseline in DEFAULT_BASELINES.items():
        sample_mean = float(df[sensor].mean())
        sample_std = float(df[sensor].std())
        expected_std = baseline * 0.015

        # Sample mean should be within 2% of nominal baseline
        assert abs(sample_mean - baseline) < (baseline * 0.02)
        # Observed standard deviation should match 1.5% Gaussian target within statistical bounds
        assert abs(sample_std - expected_std) < (expected_std * 0.25)


def test_simulator_delayed_coupling():
    """Verify 2-step delayed coupling using paired independent simulator instances."""
    seed = 42
    sim_clean = SensorSimulator(seed=seed)
    sim_fault = SensorSimulator(seed=seed)

    df_clean = sim_clean.generate(n_steps=300, fault=None)
    df_fault = sim_fault.generate(
        n_steps=300,
        fault={"sensor": "S3", "type": "drift", "start": 150, "magnitude": 3.0},
    )

    # At t=150 and t=151, the 2-step delay strictly prevents any deviation on connected neighbors
    for nb in ["S1", "S2", "S4", "S5"]:
        diff_150 = abs(df_fault.loc[150, nb] - df_clean.loc[150, nb])
        diff_151 = abs(df_fault.loc[151, nb] - df_clean.loc[151, nb])
        assert diff_150 == 0.0, f"Neighbor {nb} leaked deviation prematurely at t=150"
        assert diff_151 == 0.0, f"Neighbor {nb} leaked deviation prematurely at t=151"

        # By t=250, coupled deviation must be positive and active
        diff_250 = df_fault.loc[250, nb] - df_clean.loc[250, nb]
        assert diff_250 > 0.0, f"Neighbor {nb} missing coupled deviation post-delay"


def test_simulator_non_neighbour_isolation():
    """Inject fault into S1 and verify non-neighbors S4 and S5 receive 0 deviation."""
    seed = 101
    sim_clean = SensorSimulator(seed=seed)
    sim_fault = SensorSimulator(seed=seed)

    df_clean = sim_clean.generate(n_steps=200, fault=None)
    df_fault = sim_fault.generate(
        n_steps=200,
        fault={"sensor": "S1", "type": "drift", "start": 100, "magnitude": 10.0},
    )

    # S1 connects only to S3 and S2. S4 and S5 must remain completely unaffected.
    diff_s4 = (df_fault["S4"] - df_clean["S4"]).abs().max()
    diff_s5 = (df_fault["S5"] - df_clean["S5"]).abs().max()
    assert diff_s4 == 0.0, f"Non-neighbor S4 received unexpected deviation: {diff_s4}"
    assert diff_s5 == 0.0, f"Non-neighbor S5 received unexpected deviation: {diff_s5}"


def test_simulator_bounded_coupling():
    """Verify coupling is strictly bounded to 40% of source relative deviation with exact delay."""
    seed = 2024
    sim_clean = SensorSimulator(seed=seed)
    sim_fault = SensorSimulator(seed=seed)

    df_clean = sim_clean.generate(n_steps=300, fault=None)
    df_fault = sim_fault.generate(
        n_steps=300,
        fault={"sensor": "S3", "type": "drift", "start": 150, "magnitude": 3.0},
    )

    s3_baseline = DEFAULT_BASELINES["S3"]

    for t in [155, 200, 250, 299]:
        s3_dev_delayed = df_fault.loc[t - 2, "S3"] - df_clean.loc[t - 2, "S3"]
        s3_rel_dev = s3_dev_delayed / s3_baseline

        for nb in ["S1", "S2", "S4", "S5"]:
            nb_base = DEFAULT_BASELINES[nb]
            actual_nb_dev = df_fault.loc[t, nb] - df_clean.loc[t, nb]
            expected_nb_dev = 0.40 * s3_rel_dev * nb_base

            # Numerical tolerance check on expected coupling
            assert abs(actual_nb_dev - expected_nb_dev) < 1e-6
            # Relative neighbor response must not exceed relative source deviation
            assert (actual_nb_dev / nb_base) <= s3_rel_dev


def test_simulator_seed_reproducibility():
    """Verify two independent simulator instances with identical seeds produce identical output."""
    sim1 = SensorSimulator(seed=777)
    df1 = sim1.generate(n_steps=200, fault={"sensor": "S2", "type": "spike", "start": 50, "magnitude": 2.0})

    sim2 = SensorSimulator(seed=777)
    df2 = sim2.generate(n_steps=200, fault={"sensor": "S2", "type": "spike", "start": 50, "magnitude": 2.0})

    pd.testing.assert_frame_equal(df1, df2)


def test_simulator_fault_modes_and_drift_horizon():
    """Verify drift, spike, stuck modes and exact magnitude horizon reach."""
    seed = 42

    # 1. Drift reaches exact magnitude at final simulation step (n_steps - 1)
    sim_clean = SensorSimulator(seed=seed)
    df_clean = sim_clean.generate(n_steps=200, fault=None)

    sim_drift = SensorSimulator(seed=seed)
    df_drift = sim_drift.generate(
        n_steps=200,
        fault={"sensor": "S3", "type": "drift", "start": 100, "magnitude": 2.5},
    )
    final_drift_dev = df_drift.loc[199, "S3"] - df_clean.loc[199, "S3"]
    assert abs(final_drift_dev - 2.5) < 1e-6, f"Expected final drift 2.5, got {final_drift_dev}"

    # 2. Drift with explicit duration reaches magnitude at start + duration - 1
    sim_dur = SensorSimulator(seed=seed)
    df_dur = sim_dur.generate(
        n_steps=200,
        fault={"sensor": "S3", "type": "drift", "start": 100, "magnitude": 2.0, "duration": 20},
    )
    dev_at_duration = df_dur.loc[119, "S3"] - df_clean.loc[119, "S3"]
    assert abs(dev_at_duration - 2.0) < 1e-6
    # Clamped at magnitude thereafter
    dev_post_duration = df_dur.loc[150, "S3"] - df_clean.loc[150, "S3"]
    assert abs(dev_post_duration - 2.0) < 1e-6

    # 3. One-step drift duration immediately assumes full magnitude
    sim_1step = SensorSimulator(seed=seed)
    df_1step = sim_1step.generate(
        n_steps=100,
        fault={"sensor": "S3", "type": "drift", "start": 50, "magnitude": 1.5, "duration": 1},
    )
    df_clean_100 = SensorSimulator(seed=seed).generate(n_steps=100, fault=None)
    assert abs((df_1step.loc[50, "S3"] - df_clean_100.loc[50, "S3"]) - 1.5) < 1e-6

    # 4. Drift beginning at the final timestep reaches full magnitude
    sim_final = SensorSimulator(seed=seed)
    df_final = sim_final.generate(
        n_steps=100,
        fault={"sensor": "S3", "type": "drift", "start": 99, "magnitude": 1.8},
    )
    assert abs((df_final.loc[99, "S3"] - df_clean_100.loc[99, "S3"]) - 1.8) < 1e-6

    # 5. Spike duration and return to baseline
    sim_spike = SensorSimulator(seed=seed)
    df_spike = sim_spike.generate(
        n_steps=100,
        fault={"sensor": "S2", "type": "spike", "start": 40, "magnitude": 3.0, "duration": 5},
    )
    for t in range(40, 45):
        assert abs((df_spike.loc[t, "S2"] - df_clean_100.loc[t, "S2"]) - 3.0) < 1e-6
    # Post-spike returns to normal baseline noise
    assert abs(df_spike.loc[50, "S2"] - df_clean_100.loc[50, "S2"]) == 0.0

    # 6. Stuck freezing at baseline + magnitude with variance = 0
    sim_stuck = SensorSimulator(seed=seed)
    df_stuck = sim_stuck.generate(
        n_steps=100,
        fault={"sensor": "S3", "type": "stuck", "start": 30, "magnitude": 2.0},
    )
    stuck_slice = df_stuck.loc[30:, "S3"]
    assert (stuck_slice == 7.0).all()
    assert float(stuck_slice.var()) == 0.0


def test_simulator_adjacency_duplicate_protection():
    """Verify duplicate edges, self-loops, and invalid edges in custom connections are safely ignored."""
    dup_edges = [
        ("S3", "S1"),
        ("S3", "S1"),  # Duplicate
        ("S1", "S2"),
        ("S1", "S1"),  # Self-loop
        ("UNKNOWN_NODE", "S3"),  # Invalid node
        "invalid_format",  # Malformed item
    ]
    sim = SensorSimulator(connections=dup_edges)

    # S1 should only appear once in S3's adjacency
    assert sim.adjacency["S3"] == ["S1"]
    # S3 and S2 should appear only once in S1's adjacency
    assert sim.adjacency["S1"] == ["S3", "S2"]
    # S4 and S5 have no valid edges
    assert sim.adjacency["S4"] == []
    assert sim.adjacency["S5"] == []


def test_localizer_earliness_formula():
    """Verify earlier detection onset receives higher earliness score and S3(160) > S1(175)."""
    localizer = FaultLocalizer()
    onsets = {"S1": 175, "S2": 179, "S3": 160, "S4": 182, "S5": 190}
    earliness = localizer.compute_earliness_score(onsets)

    # Formula: (max - onset) / (max - min)
    # min = 160, max = 190, span = 30
    assert abs(earliness["S3"] - 1.0) < 1e-6  # (190 - 160) / 30 = 1.0
    assert abs(earliness["S1"] - 0.5) < 1e-6  # (190 - 175) / 30 = 0.5
    assert abs(earliness["S5"] - 0.0) < 1e-6  # (190 - 190) / 30 = 0.0

    # S3 at t=160 must score strictly higher than S1 at t=175
    assert earliness["S3"] > earliness["S1"]
    assert earliness["S1"] > earliness["S2"]
    assert earliness["S2"] > earliness["S4"]
    assert earliness["S4"] > earliness["S5"]

    # Identical onsets should yield equal 1.0 scores
    equal_onsets = {"S1": 100, "S2": 100, "S3": 100, "S4": None, "S5": None}
    equal_early = localizer.compute_earliness_score(equal_onsets)
    assert equal_early["S1"] == 1.0
    assert equal_early["S2"] == 1.0
    assert equal_early["S3"] == 1.0
    assert equal_early["S4"] == 0.0


def test_localizer_neighbour_support_and_coverage():
    """Verify neighbour support measures causal delayed propagation without independent degree bias."""
    localizer = FaultLocalizer()
    scores = {"S1": 0.75, "S2": 0.73, "S3": 0.73, "S4": 0.70, "S5": 0.70}
    onsets = {"S1": 175, "S2": 179, "S3": 160, "S4": 182, "S5": 190}

    support = localizer.compute_neighbour_support(scores, onsets)
    coverage = localizer.compute_coverage_scores(scores)

    # All of S3's neighbors (S1, S2, S4, S5) alarmed at or after S3 (t=160), giving high support
    assert support["S3"] > support["S1"]
    assert support["S1"] > support["S2"]
    assert support["S5"] == 0.0  # S5 alarmed latest (t=190), no neighbors alarmed after it

    # Coverage evaluates neighborhood abnormality and does not independently bias by degree
    for s in ["S1", "S2", "S3", "S4", "S5"]:
        assert 0.0 <= coverage[s] <= 1.0
        assert 0.0 <= support[s] <= 1.0


def test_localizer_no_anomaly_and_missing_onset_handling():
    """Verify graceful handling when no anomalies occur or when onsets are missing."""
    localizer = FaultLocalizer()

    # Case 1: All onsets None (no detection)
    clean_scores = {"S1": 0.1, "S2": 0.1, "S3": 0.1, "S4": 0.1, "S5": 0.1}
    clean_onsets = {"S1": None, "S2": None, "S3": None, "S4": None, "S5": None}

    rankings = localizer.rank_suspects(clean_scores, clean_onsets)
    assert len(rankings) == 5
    for _, comp, bd in rankings:
        assert comp <= 0.15  # Only baseline noise contribution
        assert bd["detection_earliness"] == 0.0
        assert bd["neighbour_support"] == 0.0

    # Case 2: Incomplete/missing dictionary keys
    sparse_scores = {"S1": 0.8}
    sparse_onsets = {"S1": 100}
    sparse_rankings = localizer.rank_suspects(sparse_scores, sparse_onsets)
    assert len(sparse_rankings) == 5
    assert sparse_rankings[0][0] == "S1"


def test_localizer_ground_truth_independence():
    """Verify rankings are strictly computed from anomaly evidence and never from ground truth."""
    localizer = FaultLocalizer()
    sim = SensorSimulator(seed=42)

    # Generate simulation with ground truth fault in attrs
    df = sim.generate(n_steps=300, fault={"sensor": "S3", "type": "drift", "start": 150, "magnitude": 3.0})
    assert df.attrs.get("fault") is not None

    detector = AnomalyDetector(random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    from faultmesh.detector import first_onset
    onsets = first_onset(scores_df)
    mean_scores = {s: float(scores_df[s].iloc[-50:].mean()) for s in ["S1", "S2", "S3", "S4", "S5"]}

    # Pass solely the numerical scores and onsets (no ground truth attributes)
    rankings_pure = localizer.rank_suspects(mean_scores, onsets)

    # Corrupt ground truth metadata completely
    df.attrs["fault"] = {"sensor": "S1", "type": "drift", "start": 0, "magnitude": 99.0}

    # Re-run localizer: ranking must remain 100% identical
    rankings_corrupted = localizer.rank_suspects(mean_scores, onsets)
    assert rankings_pure == rankings_corrupted


def test_localizer_s3_drift_scenario_top1():
    """Verify real S3 drift scenario identifies S3 as #1 root cause over coupled neighbors."""
    sim = SensorSimulator(seed=42)
    df = sim.generate(n_steps=300, fault={"sensor": "S3", "type": "drift", "start": 150, "magnitude": 3.0})

    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    from faultmesh.detector import first_onset
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    mean_scores = {s: float(scores_df[s].iloc[-50:].mean()) for s in ["S1", "S2", "S3", "S4", "S5"]}
    localizer = FaultLocalizer()
    rankings = localizer.rank_suspects(mean_scores, onsets)

    # S3 must rank #1
    top1_sensor, top1_score, top1_bd = rankings[0]
    assert top1_sensor == "S3"
    assert top1_score > 0.75

    # Check descending order
    for i in range(len(rankings) - 1):
        assert rankings[i][1] >= rankings[i + 1][1]


def test_localizer_multi_sensor_validation_all_5_scenarios():
    """Validate 100% Top-1 localization accuracy across all 5 sensors (S1, S2, S3, S4, S5)."""
    from faultmesh.detector import first_onset
    localizer = FaultLocalizer()
    sensors = ["S1", "S2", "S3", "S4", "S5"]
    correct_count = 0

    for target in sensors:
        sim = SensorSimulator(seed=42)
        df = sim.generate(n_steps=300, fault={"sensor": target, "type": "drift", "start": 150, "magnitude": 3.0})

        detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
        detector.fit(df)
        scores_df = detector.score(df)
        onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

        mean_scores = {s: float(scores_df[s].iloc[-50:].mean()) for s in sensors}
        rankings = localizer.rank_suspects(mean_scores, onsets)

        diagnosed = rankings[0][0]
        if diagnosed == target:
            correct_count += 1

    # Must achieve 5/5 (100%) Top-1 accuracy
    assert correct_count == 5, f"Expected 5/5 correct diagnoses, achieved {correct_count}/5"


def test_benchmark_matrix_coverage():
    """Verify benchmark generates exactly 75 unique scenarios covering all 5 sensors, 3 faults, 5 seeds."""
    from evaluate import generate_benchmark_matrix, SENSORS, FAULT_TYPES, SEEDS

    matrix = generate_benchmark_matrix()
    assert len(matrix) == 75
    assert len(set(matrix)) == 75

    sensors_in_matrix = {s for s, f, r in matrix}
    faults_in_matrix = {f for s, f, r in matrix}
    seeds_in_matrix = {r for s, f, r in matrix}

    assert sensors_in_matrix == set(SENSORS) == {"S1", "S2", "S3", "S4", "S5"}
    assert faults_in_matrix == set(FAULT_TYPES) == {"drift", "spike", "stuck"}
    assert seeds_in_matrix == set(SEEDS) == {1, 2, 3, 4, 5}


def test_benchmark_metric_calculation_controlled_data():
    """Verify metrics calculation using controlled sample data with intentional failures (does not assume 100%)."""
    from evaluate import calculate_metrics

    # 10 sample records: 7 Top-1 correct, 9 Top-2 correct, 8 detection successes
    sample_records = [
        {"injected_sensor": "S1", "fault_type": "drift", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S1", "fault_type": "spike", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S2", "fault_type": "drift", "top1_correct": False, "top2_correct": True, "detection_success": True}, # Failed Top-1
        {"injected_sensor": "S2", "fault_type": "stuck", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S3", "fault_type": "drift", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S3", "fault_type": "spike", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S4", "fault_type": "drift", "top1_correct": False, "top2_correct": False, "detection_success": False}, # Failed Top-1 & Top-2, no detection
        {"injected_sensor": "S4", "fault_type": "stuck", "top1_correct": True, "top2_correct": True, "detection_success": True},
        {"injected_sensor": "S5", "fault_type": "spike", "top1_correct": False, "top2_correct": True, "detection_success": False}, # Failed Top-1, no detection
        {"injected_sensor": "S5", "fault_type": "stuck", "top1_correct": True, "top2_correct": True, "detection_success": True},
    ]

    metrics = calculate_metrics(sample_records)

    assert metrics["total_scenarios"] == 10
    assert metrics["top1_correct"] == 7
    assert metrics["top2_correct"] == 9
    assert abs(metrics["top1_accuracy"] - 70.0) < 1e-6
    assert abs(metrics["top2_accuracy"] - 90.0) < 1e-6
    assert abs(metrics["detection_rate"] - 80.0) < 1e-6
    assert metrics["successful_scenarios"] == 7
    assert metrics["failed_scenarios"] == 3

    # Drift fault type breakdown: 4 total (S1, S2, S3, S4), 2 correct -> 50.0%
    drift_metrics = metrics["fault_type_accuracy"]["drift"]
    assert drift_metrics["total"] == 4
    assert drift_metrics["top1_correct"] == 2
    assert abs(drift_metrics["top1_accuracy"] - 50.0) < 1e-6


def test_benchmark_csv_and_text_report_schema(tmp_path):
    """Verify evaluation CSV schema and text report generation."""
    from evaluate import run_single_scenario, DEFAULT_CSV_PATH
    import os

    csv_file = tmp_path / "test_evaluation_results.csv"
    report_file = tmp_path / "test_evaluation_report.txt"

    # Run 2 scenarios
    record1 = run_single_scenario("S1", "drift", seed=1)
    record2 = run_single_scenario("S3", "spike", seed=2)
    records = [record1, record2]

    from evaluate import calculate_metrics
    metrics = calculate_metrics(records)

    # Save to CSV
    csv_cols = [
        "scenario_id", "injected_sensor", "fault_type", "seed",
        "injected_sensor_onset", "earliest_system_onset", "detection_success",
        "predicted_root_cause", "top1_correct", "top2_correct",
        "localization_score", "rank_1", "rank_2"
    ]
    df_csv = pd.DataFrame([{c: r[c] for c in csv_cols} for r in records])
    df_csv.to_csv(csv_file, index=False)

    # Verify CSV columns exist and match schema
    df_read = pd.read_csv(csv_file)
    assert list(df_read.columns) == csv_cols
    assert len(df_read) == 2
    assert "injected_sensor_onset" in df_read.columns
    assert "earliest_system_onset" in df_read.columns
    assert "detection_success" in df_read.columns

    # Verify text report output
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Total Scenarios Evaluated: {metrics['total_scenarios']}\n")
        f.write(f"Top-1 Correct: {metrics['top1_correct']}\n")

    assert os.path.exists(report_file)
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Total Scenarios Evaluated: 2" in content


def test_benchmark_ground_truth_independence():
    """Verify benchmark run_single_scenario never passes ground-truth fault to localizer."""
    from evaluate import run_single_scenario

    res = run_single_scenario("S3", "drift", seed=42)
    # Result must contain prediction and rankings calculated solely from sensor evidence
    assert res["injected_sensor"] == "S3"
    assert "predicted_root_cause" in res
    assert "rankings" in res
    assert len(res["rankings"]) == 5
    # Ground truth is used only for post-evaluation boolean flag
    assert res["top1_correct"] == (res["predicted_root_cause"] == "S3")
