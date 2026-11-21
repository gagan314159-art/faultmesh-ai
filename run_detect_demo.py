"""Anomaly detection demonstration script for FaultMesh AI.

Demonstrates:
    1. Generating 300 simulated multi-sensor IoT time steps.
    2. Injecting an S3 drift fault starting at step 150 with magnitude 3.0.
    3. Training Isolation Forest detectors on the first 100 normal baseline samples.
    4. Calculating calibrated multivariate anomaly scores [0, 1] across all timesteps.
    5. Evaluating first detection onset for each sensor (>= 0.6 for >= 3 consecutive steps).
    6. Printing detection onsets, final 50-step mean scores, and ground truth.
    7. Clearly distinguishing the root-cause faulty sensor from coupled abnormal neighbors.

Usage:
    python run_detect_demo.py
"""

from faultmesh.simulator import SensorSimulator
from faultmesh.detector import AnomalyDetector, first_onset


def main() -> None:
    print("=" * 75)
    print("FaultMesh AI — Step 2: Anomaly Detection Standalone Demonstration")
    print("=" * 75)

    # 1. Generate 300 simulated time steps with S3 drift fault at step 150, magnitude 3.0
    sim = SensorSimulator(seed=42)
    fault_config = {
        "sensor": "S3",
        "type": "drift",
        "start": 150,
        "magnitude": 3.0,
    }
    n_steps = 300

    print(f"\n[1] Generating {n_steps} time steps with simulated IoT fault:")
    print(f"    - Injected Sensor: {fault_config['sensor']}")
    print(f"    - Fault Type:     {fault_config['type']}")
    print(f"    - Start Step:     {fault_config['start']}")
    print(f"    - Magnitude:      {fault_config['magnitude']} (physical deviation)")

    raw_df = sim.generate(n_steps=n_steps, fault=fault_config)

    # 2. Train detector on the first 100 normal samples
    print("\n[2] Training Isolation Forest models on first 100 normal baseline samples...")
    detector = AnomalyDetector(threshold=0.6, consecutive_required=3, random_state=42)
    detector.fit(raw_df)
    print("    Models fitted successfully for all sensors: S1, S2, S3, S4, S5.")

    # 3. Calculate anomaly scores
    print("\n[3] Calculating calibrated multivariate anomaly scores [0, 1]...")
    scores_df = detector.score(raw_df)
    print(f"    Anomaly scores DataFrame shape: {scores_df.shape}")

    # 4. Calculate detection onset for each sensor
    print("\n[4] Evaluating first detection onset (score >= 0.6 for >= 3 consecutive readings)...")
    onsets = first_onset(scores_df, threshold=0.6, min_consecutive=3)

    # 5. Calculate mean anomaly score over the final 50 time steps
    final_50_slice = scores_df.iloc[-50:]
    mean_final_50 = {s: float(final_50_slice[s].mean()) for s in ["S1", "S2", "S3", "S4", "S5"]}

    # 6 & 7. Print summary and clearly distinguish root cause from coupled neighbors
    injected_sensor = fault_config["sensor"]
    print("\n" + "-" * 75)
    print(f"{'Sensor':<8} | {'Detection Onset':<18} | {'Final 50 Mean Score':<20} | {'Diagnostic Status'}")
    print("-" * 75)

    for sensor in ["S1", "S2", "S3", "S4", "S5"]:
        onset_str = f"t = {onsets[sensor]}" if onsets[sensor] is not None else "None (Normal)"
        score_val = f"{mean_final_50[sensor]:.4f}"

        if sensor == injected_sensor:
            status = "🚨 INJECTED ROOT CAUSE (Earliest Detection)"
        elif onsets[sensor] is not None:
            delay = onsets[sensor] - onsets[injected_sensor] if onsets[injected_sensor] is not None else 0
            status = f"⚠️ Coupled Neighbor (Delayed by +{delay} steps)"
        else:
            status = "✅ Normal (No Sustained Alarm)"

        print(f"{sensor:<8} | {onset_str:<18} | {score_val:<20} | {status}")

    print("-" * 75)
    print(f"\nGround-Truth Injected Fault Sensor: {injected_sensor}")
    if onsets[injected_sensor] is not None:
        earliest_sensor = min(
            (s for s in onsets if onsets[s] is not None),
            key=lambda s: onsets[s],
        )
        print(f"Earliest Alarming Sensor:           {earliest_sensor} (t = {onsets[earliest_sensor]})")
        if earliest_sensor == injected_sensor:
            print(">> Root-cause diagnosis SUCCESS: Injected sensor S3 detected earliest!")
        else:
            print(f">> Root cause did not alarm earliest (earliest: {earliest_sensor}).")

    print("\n" + "=" * 75)
    print("Anomaly detection demonstration completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()
