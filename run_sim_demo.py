"""Sensor Simulator demonstration script for FaultMesh AI.

Demonstrates:
    1. Initializing the SensorSimulator with a fixed random seed.
    2. Generating 300 time steps of multi-sensor IoT readings.
    3. Injecting an S3 drift fault starting at step 150 with magnitude 3.0.
    4. Displaying the final 10 rows of sensor readings.
    5. Showing the ground-truth fault information stored in df.attrs['fault'].
    6. Persisting generated readings to SQLite storage (data/faultmesh.db).

Usage:
    python run_sim_demo.py
"""

from faultmesh.simulator import SensorSimulator
from faultmesh.storage import save_readings


def main() -> None:
    print("=" * 70)
    print("FaultMesh AI — Step 1: Sensor Simulator & SQLite Storage Demo")
    print("=" * 70)

    # 1. Initialize SensorSimulator
    seed = 42
    print(f"\n[1] Initializing SensorSimulator (seed={seed})...")
    sim = SensorSimulator(seed=seed)

    # 2 & 3. Define S3 drift fault and generate 300 time steps
    fault_config = {
        "sensor": "S3",
        "type": "drift",
        "start": 150,
        "magnitude": 3.0,
    }
    n_steps = 300
    print(f"[2] Generating {n_steps} time steps with injected fault:")
    print(f"    Fault config: {fault_config}")

    df = sim.generate(n_steps=n_steps, fault=fault_config)

    # 4. Print final 10 rows
    print("\n[3] Final 10 rows of simulated readings (df.tail(10)):")
    print(df.tail(10).to_string(index=False))

    # 5. Display ground-truth fault information
    print("\n[4] Ground-truth fault metadata (df.attrs['fault']):")
    print(f"    {df.attrs.get('fault')}")

    # 6. Save readings to SQLite
    print("\n[5] Persisting readings to SQLite database (data/faultmesh.db)...")
    inserted_rows = save_readings(df)
    print(f"    Successfully stored {inserted_rows} reading records into 'readings' table.")

    print("\n" + "=" * 70)
    print("Demo execution finished successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
