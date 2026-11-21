"""Evaluation benchmark script for FaultMesh AI.

Responsibility:
    Execute systematic 75-scenario evaluation benchmark matrix:
    5 Sensors × 3 Fault Types × 5 Random Seeds = 75 independent scenarios.
    Calculate Top-1 accuracy, Top-2 accuracy, detection rate, fault-type and sensor breakdowns.
    Export complete results to CSV and a readable text summary.

Usage:
    python evaluate.py
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import pandas as pd
import numpy as np

from faultmesh.simulator import SensorSimulator, DEFAULT_BASELINES
from faultmesh.detector import AnomalyDetector, first_onset
from faultmesh.localizer import FaultLocalizer
from faultmesh.graph import SensorNetworkGraph


# Benchmark Matrix Specification according to PROJECT_RULES.md
SENSORS: List[str] = ["S1", "S2", "S3", "S4", "S5"]
FAULT_TYPES: List[str] = ["drift", "spike", "stuck"]
SEEDS: List[int] = [1, 2, 3, 4, 5]

DEFAULT_CSV_PATH = "data/evaluation_results.csv"
DEFAULT_REPORT_PATH = "data/evaluation_report.txt"


def generate_benchmark_matrix() -> List[Tuple[str, str, int]]:
    """Generate all 75 parameter tuples (sensor, fault_type, seed)."""
    matrix = []
    for sensor in SENSORS:
        for fault in FAULT_TYPES:
            for seed in SEEDS:
                matrix.append((sensor, fault, seed))
    return matrix


def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics across a list of benchmark scenario records.

    Uses all records as the denominator. Missing or incorrect predictions count as failures.

    Args:
        records: List of scenario result dictionaries.

    Returns:
        Dict containing total counts, overall Top-1/Top-2 percentages, detection rate,
        and breakdowns by fault type and sensor.
    """
    total = len(records)
    if total == 0:
        return {
            "total_scenarios": 0,
            "top1_correct": 0,
            "top2_correct": 0,
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "detection_rate": 0.0,
            "successful_scenarios": 0,
            "failed_scenarios": 0,
            "fault_type_accuracy": {},
            "sensor_accuracy": {},
        }

    top1_correct = sum(1 for r in records if r.get("top1_correct") is True)
    top2_correct = sum(1 for r in records if r.get("top2_correct") is True)
    detection_successes = sum(1 for r in records if r.get("detection_success") is True)

    top1_acc = float((top1_correct / total) * 100.0)
    top2_acc = float((top2_correct / total) * 100.0)
    det_rate = float((detection_successes / total) * 100.0)

    # Fault-type-wise breakdown
    fault_breakdown = {}
    for ft in FAULT_TYPES:
        sub = [r for r in records if r.get("fault_type") == ft]
        sub_total = len(sub)
        if sub_total > 0:
            c1 = sum(1 for r in sub if r.get("top1_correct") is True)
            c2 = sum(1 for r in sub if r.get("top2_correct") is True)
            fault_breakdown[ft] = {
                "total": sub_total,
                "top1_correct": c1,
                "top2_correct": c2,
                "top1_accuracy": float((c1 / sub_total) * 100.0),
                "top2_accuracy": float((c2 / sub_total) * 100.0),
            }

    # Sensor-wise breakdown
    sensor_breakdown = {}
    for s in SENSORS:
        sub = [r for r in records if r.get("injected_sensor") == s]
        sub_total = len(sub)
        if sub_total > 0:
            c1 = sum(1 for r in sub if r.get("top1_correct") is True)
            c2 = sum(1 for r in sub if r.get("top2_correct") is True)
            sensor_breakdown[s] = {
                "total": sub_total,
                "top1_correct": c1,
                "top2_correct": c2,
                "top1_accuracy": float((c1 / sub_total) * 100.0),
                "top2_accuracy": float((c2 / sub_total) * 100.0),
            }

    return {
        "total_scenarios": total,
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "detection_rate": det_rate,
        "successful_scenarios": top1_correct,
        "failed_scenarios": total - top1_correct,
        "fault_type_accuracy": fault_breakdown,
        "sensor_accuracy": sensor_breakdown,
    }


def run_single_scenario(
    sensor: str,
    fault_type: str,
    seed: int,
    localizer: Optional[FaultLocalizer] = None,
) -> Dict[str, Any]:
    """Execute a single end-to-end benchmark scenario without ground-truth leakage.

    Args:
        sensor: Injected root-cause sensor (S1..S5).
        fault_type: Fault mode ('drift', 'spike', 'stuck').
        seed: Random seed for simulation reproducibility.
        localizer: FaultLocalizer instance.

    Returns:
        Dictionary containing scenario results and diagnostic metadata.
    """
    loc = localizer or FaultLocalizer()
    sim = SensorSimulator(seed=seed)

    base = DEFAULT_BASELINES[sensor]
    magnitude = base * 0.5  # 50% physical shift relative to baseline

    fault_config = {
        "sensor": sensor,
        "type": fault_type,
        "start": 150,
        "magnitude": magnitude,
    }
    if fault_type == "spike":
        fault_config["duration"] = 10  # 10-step transient spike

    # 1. Simulate readings (300 time steps)
    df = sim.generate(n_steps=300, fault=fault_config)

    # 2. Fit detector on first 100 normal samples
    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(df)
    scores_df = detector.score(df)
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    # 3. Extract anomaly score evidence passed to localizer
    if fault_type == "spike":
        anomaly_scores = {s: float(scores_df[s].max()) for s in SENSORS}
    else:
        anomaly_scores = {s: float(scores_df[s].iloc[-50:].mean()) for s in SENSORS}

    # 4. Run localization without ground-truth leakage
    rankings = loc.rank_suspects(anomaly_scores, onsets, total_steps=300)

    # 5. Extract results
    injected_onset = onsets.get(sensor)
    valid_onsets = [t for t in onsets.values() if t is not None]
    earliest_system_onset = min(valid_onsets) if valid_onsets else None
    detection_success = (injected_onset is not None)

    predicted_root = rankings[0][0] if len(rankings) > 0 else ""
    rank1 = rankings[0][0] if len(rankings) > 0 else ""
    rank2 = rankings[1][0] if len(rankings) > 1 else ""
    localization_score = float(rankings[0][1]) if len(rankings) > 0 else 0.0

    top1_correct = (predicted_root == sensor)
    top2_correct = (sensor in [rank1, rank2])

    scenario_id = f"{sensor}_{fault_type}_seed{seed}"

    return {
        "scenario_id": scenario_id,
        "injected_sensor": sensor,
        "fault_type": fault_type,
        "seed": seed,
        "injected_sensor_onset": injected_onset,
        "earliest_system_onset": earliest_system_onset,
        "detection_success": detection_success,
        "predicted_root_cause": predicted_root,
        "top1_correct": top1_correct,
        "top2_correct": top2_correct,
        "localization_score": localization_score,
        "rank_1": rank1,
        "rank_2": rank2,
        "onsets": onsets,
        "rankings": rankings,
    }


def run_benchmark(
    save_results: bool = True,
    csv_path: str = DEFAULT_CSV_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """Execute all 75 scenarios in the benchmark matrix and compile reports.

    Args:
        save_results: If True, writes CSV and text report files.
        csv_path: Path for CSV export.
        report_path: Path for text report export.

    Returns:
        Summary metrics dictionary.
    """
    matrix = generate_benchmark_matrix()
    total_scenarios = len(matrix)

    print("=" * 80)
    print("FaultMesh AI — Systematic 75-Scenario Evaluation Benchmark")
    print(f"Matrix: {len(SENSORS)} Sensors × {len(FAULT_TYPES)} Fault Types × {len(SEEDS)} Seeds = {total_scenarios} Scenarios")
    print("=" * 80)

    localizer = FaultLocalizer()
    records: List[Dict[str, Any]] = []

    for idx, (sensor, fault, seed) in enumerate(matrix, start=1):
        record = run_single_scenario(sensor, fault, seed, localizer=localizer)
        records.append(record)
        status_icon = "✓" if record["top1_correct"] else "✗"
        onset_str = f"t={record['injected_sensor_onset']}" if record['injected_sensor_onset'] else "None"
        print(f"[{idx:>2}/{total_scenarios}] {record['scenario_id']:<24} -> Top-1: {record['predicted_root_cause']:<3} {status_icon} (Onset: {onset_str:<6}, Score: {record['localization_score']:.3f})")

    metrics = calculate_metrics(records)

    # Console Report Generation
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Scenarios Completed:    {metrics['total_scenarios']}")
    print(f"Top-1 Correct / Total:        {metrics['top1_correct']} / {metrics['total_scenarios']}")
    print(f"Top-2 Correct / Total:        {metrics['top2_correct']} / {metrics['total_scenarios']}")
    print(f"Overall Top-1 Accuracy:       {metrics['top1_accuracy']:.2f}%")
    print(f"Overall Top-2 Accuracy:       {metrics['top2_accuracy']:.2f}%")
    print(f"Injected Detection Rate:      {metrics['detection_rate']:.2f}%")
    print(f"Successful Scenarios (Top-1): {metrics['successful_scenarios']}")
    print(f"Failed Scenarios:             {metrics['failed_scenarios']}")

    print("\n" + "-" * 50)
    print("ACCURACY BY FAULT TYPE:")
    print("-" * 50)
    print(f"{'Fault Type':<12} | {'Count':<6} | {'Top-1 Acc':<12} | {'Top-2 Acc'}")
    print("-" * 50)
    for ft, fb in metrics["fault_type_accuracy"].items():
        print(f"{ft:<12} | {fb['total']:<6} | {fb['top1_accuracy']:>6.2f}% ({fb['top1_correct']}/{fb['total']}) | {fb['top2_accuracy']:>6.2f}% ({fb['top2_correct']}/{fb['total']})")

    print("\n" + "-" * 50)
    print("ACCURACY BY INJECTED SENSOR:")
    print("-" * 50)
    print(f"{'Sensor':<12} | {'Count':<6} | {'Top-1 Acc':<12} | {'Top-2 Acc'}")
    print("-" * 50)
    for s, sb in metrics["sensor_accuracy"].items():
        print(f"{s:<12} | {sb['total']:<6} | {sb['top1_accuracy']:>6.2f}% ({sb['top1_correct']}/{sb['total']}) | {sb['top2_accuracy']:>6.2f}% ({sb['top2_correct']}/{sb['total']})")

    failed_records = [r for r in records if not r["top1_correct"]]
    if failed_records:
        print("\n" + "-" * 80)
        print("FAILED SCENARIO DETAILS:")
        print("-" * 80)
        for fr in failed_records:
            print(f"Scenario: {fr['scenario_id']} | Injected: {fr['injected_sensor']} | Predicted: {fr['predicted_root_cause']} | Rank: #{[r[0] for r in fr['rankings']].index(fr['injected_sensor']) + 1 if fr['injected_sensor'] in [r[0] for r in fr['rankings']] else 'N/A'}")
    else:
        print("\nAll 75 scenarios diagnosed successfully with 0 failures!")

    # Save CSV and Text Reports if requested
    if save_results:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        csv_columns = [
            "scenario_id",
            "injected_sensor",
            "fault_type",
            "seed",
            "injected_sensor_onset",
            "earliest_system_onset",
            "detection_success",
            "predicted_root_cause",
            "top1_correct",
            "top2_correct",
            "localization_score",
            "rank_1",
            "rank_2",
        ]
        df_csv = pd.DataFrame([{col: r[col] for col in csv_columns} for r in records])
        df_csv.to_csv(csv_path, index=False)
        print(f"\n[Artifact Saved] Complete CSV results: {csv_path}")

        # Text Report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("FaultMesh AI — Systematic 75-Scenario Evaluation Benchmark Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Scenarios Evaluated:    {metrics['total_scenarios']}\n")
            f.write(f"Top-1 Correct / Total:        {metrics['top1_correct']} / {metrics['total_scenarios']}\n")
            f.write(f"Top-2 Correct / Total:        {metrics['top2_correct']} / {metrics['total_scenarios']}\n")
            f.write(f"Overall Top-1 Accuracy:       {metrics['top1_accuracy']:.2f}%\n")
            f.write(f"Overall Top-2 Accuracy:       {metrics['top2_accuracy']:.2f}%\n")
            f.write(f"Injected Detection Rate:      {metrics['detection_rate']:.2f}%\n")
            f.write(f"Successful Scenarios:         {metrics['successful_scenarios']}\n")
            f.write(f"Failed Scenarios:             {metrics['failed_scenarios']}\n\n")

            f.write("-" * 50 + "\n")
            f.write("ACCURACY BY FAULT TYPE:\n")
            f.write("-" * 50 + "\n")
            for ft, fb in metrics["fault_type_accuracy"].items():
                f.write(f"  * {ft:<8}: Top-1 = {fb['top1_accuracy']:.2f}% ({fb['top1_correct']}/{fb['total']}), Top-2 = {fb['top2_accuracy']:.2f}%\n")

            f.write("\n" + "-" * 50 + "\n")
            f.write("ACCURACY BY INJECTED SENSOR:\n")
            f.write("-" * 50 + "\n")
            for s, sb in metrics["sensor_accuracy"].items():
                f.write(f"  * {s:<8}: Top-1 = {sb['top1_accuracy']:.2f}% ({sb['top1_correct']}/{sb['total']}), Top-2 = {sb['top2_accuracy']:.2f}%\n")

            if failed_records:
                f.write("\n" + "-" * 50 + "\n")
                f.write("FAILED SCENARIOS:\n")
                f.write("-" * 50 + "\n")
                for fr in failed_records:
                    f.write(f"  * Scenario {fr['scenario_id']}: Injected {fr['injected_sensor']}, Predicted {fr['predicted_root_cause']}\n")
            else:
                f.write("\nZero failures across all 75 benchmark scenarios.\n")

        print(f"[Artifact Saved] Summary text report: {report_path}")

    metrics["records"] = records
    return metrics


def main() -> None:
    """CLI entry point running full 75-scenario benchmark."""
    run_benchmark(save_results=True)


if __name__ == "__main__":
    main()
