"""Fault localization demonstration script for FaultMesh AI.

Demonstrates:
    1. Generating 300 steps of simulated IoT mesh readings with an S3 drift fault at step 150.
    2. Training the Isolation Forest detector on the first 100 normal baseline samples.
    3. Calculating calibrated multivariate anomaly scores [0, 1] across all timesteps.
    4. Extracting exact detection onsets (>= 0.6 for >= 3 consecutive timesteps).
    5. Evaluating multi-criteria root-cause ranking (35% own, 30% earliness, 20% support, 15% coverage).
    6. Reporting complete component breakdowns, diagnostic explanations, and post-localization accuracy.

Usage:
    python run_localize_demo.py
"""

from faultmesh.simulator import SensorSimulator
from faultmesh.detector import AnomalyDetector, first_onset
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer


def main() -> None:
    print("=" * 80)
    print("FaultMesh AI — Step 3: Graph-Based Fault Localization Demonstration")
    print("=" * 80)

    # 1. Simulator setup (Injected ground-truth fault: S3 drift)
    sim = SensorSimulator(seed=42)
    fault_config = {
        "sensor": "S3",
        "type": "drift",
        "start": 150,
        "magnitude": 3.0,
    }
    n_steps = 300

    print("\n[1] Simulation & Injected Fault Configuration (Ground Truth):")
    print(f"    - Target Sensor:  {fault_config['sensor']}")
    print(f"    - Fault Type:     {fault_config['type']}")
    print(f"    - Start Timestep: {fault_config['start']}")
    print(f"    - Magnitude:      {fault_config['magnitude']} (physical deviation)")
    print(f"    - Total Steps:    {n_steps}")

    raw_df = sim.generate(n_steps=n_steps, fault=fault_config)

    # 2. Fit detector on first 100 normal samples and score all timesteps
    print("\n[2] Training Isolation Forest models on first 100 normal baseline samples...")
    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(raw_df)

    print("[3] Calculating actual anomaly scores across all timesteps...")
    scores_df = detector.score(raw_df)

    # 3. Calculate detection onsets
    print("[4] Calculating actual detection onset for each sensor...")
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    # TASK 1: Print exact detection onset dictionary immediately before localization
    print("\n" + "-" * 80)
    print("TASK 1 — Exact Detection Onset Dictionary (Passed Directly to Localizer):")
    print(f"    {onsets}")
    print("-" * 80)

    # TASK 3: Print final actual anomaly scores passed to localizer (final 50-step mean)
    final_50_slice = scores_df.iloc[-50:]
    anomaly_scores = {s: float(final_50_slice[s].mean()) for s in ["S1", "S2", "S3", "S4", "S5"]}
    print("\nTASK 3 — Actual Anomaly Scores (Passed to Localizer):")
    for s, score in anomaly_scores.items():
        print(f"    Sensor {s}: {score:.4f}")

    # 4. Initialize Localizer and compute components
    localizer = FaultLocalizer()

    # TASK 2: Print calculated earliness score for every sensor
    earliness = localizer.compute_earliness_score(onsets, max_timesteps=n_steps)
    print("\nTASK 2 — Calculated Detection Earliness Scores:")
    for s, early in earliness.items():
        print(f"    Sensor {s} (onset={onsets[s]}): earliness = {early:.4f}")

    # TASK 4: Print neighbour-support calculation for every sensor
    support = localizer.compute_neighbour_support(anomaly_scores, onsets)
    print("\nTASK 4 — Calculated Neighbour Support Scores (Causal Delayed Evidence):")
    for s, supp in support.items():
        print(f"    Sensor {s}: neighbour_support = {supp:.4f}")

    # TASK 5: Print graph-coverage calculation for every sensor
    coverage = localizer.compute_coverage_scores(anomaly_scores)
    print("\nTASK 5 — Calculated Graph Coverage Scores (Neighborhood Abnormality):")
    for s, cov in coverage.items():
        print(f"    Sensor {s}: graph_coverage = {cov:.4f}")

    # TASK 6: Run ranking with specified initial weights (0.35, 0.30, 0.20, 0.15)
    rankings = localizer.rank_suspects(anomaly_scores, onsets, total_steps=n_steps)

    # TASK 9: Display complete localization ranking output
    print("\n" + "=" * 80)
    print("TASK 9 — Root-Cause Localization Ranking Results:")
    print("Weights: Own (35%) | Earliness (30%) | Support (20%) | Coverage (15%)")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Sensor':<7} | {'Score':<8} | {'Own (35%)':<10} | {'Early (30%)':<12} | {'Supp (20%)':<11} | {'Cov (15%)':<10}")
    print("-" * 80)

    for rank, (sensor, composite, bd) in enumerate(rankings, start=1):
        print(
            f"#{rank:<5} | {sensor:<7} | {composite:<8.4f} | "
            f"{bd['own_anomaly']:<10.4f} | {bd['detection_earliness']:<12.4f} | "
            f"{bd['neighbour_support']:<11.4f} | {bd['coverage']:<10.4f}"
        )
        print(f"       Diagnostic: {bd['explanation']}\n")

    # Final Top-1 diagnosis result & post-ranking evaluation against ground truth
    top1_diagnosed = rankings[0][0]
    injected_sensor = fault_config["sensor"]
    print("=" * 80)
    print(f"Top Diagnosed Root Cause:         {top1_diagnosed}")
    print(f"Ground-Truth Injected Sensor:     {injected_sensor} (Evaluated after localization)")
    if top1_diagnosed == injected_sensor:
        print(">> VERIFICATION SUCCESS: S3 ranked #1 above all coupled neighbors!")
    else:
        print(f">> VERIFICATION FAILURE: Expected {injected_sensor}, got {top1_diagnosed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
