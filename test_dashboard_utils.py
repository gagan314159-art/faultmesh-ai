from dashboard.utils import (
    SimConfig,
    run_simulation,
    run_detection,
    run_localization
)

sensors = ["S1", "S2", "S3", "S4", "S5"]
fault_types = ["drift", "spike", "stuck"]

correct = 0
total = 0

print("SENSOR | FAULT | TOP-1 | RESULT")
print("-" * 45)

for sensor in sensors:
    for fault_type in fault_types:
        cfg = SimConfig(
            seed=42,
            fault_sensor=sensor,
            fault_type=fault_type,
            fault_start=150
        )

        df, fault = run_simulation(cfg, use_cache=False)
        scores, onsets = run_detection(df)
        rankings = run_localization(scores, onsets)

        top = rankings[0][0] if rankings else "NONE"
        passed = top == sensor

        correct += int(passed)
        total += 1

        result = "PASS" if passed else "FAIL"
        print(f"{sensor} | {fault_type} | {top} | {result}")

print("-" * 45)
print(f"TOP-1: {correct}/{total} ({100 * correct / total:.1f}%)")
