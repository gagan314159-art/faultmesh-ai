"""Simulator module for FaultMesh AI.

Responsibility:
    Generate realistic multi-sensor IoT time-series data with Gaussian baseline noise,
    inject physical/electrical faults (drift, spike, stuck), and propagate neighbor
    coupling dynamics across the mesh network with time delay.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# Default baseline sensor configurations according to PROJECT_RULES.md
DEFAULT_BASELINES: Dict[str, float] = {
    "S1": 60.0,  # Temperature (°C)
    "S2": 2.0,   # Vibration (mm/s)
    "S3": 5.0,   # Pressure (bar)
    "S4": 30.0,  # Flow (L/min)
    "S5": 10.0,  # Current (A)
}

# Default undirected network topology connections defined in PROJECT_RULES.md
DEFAULT_CONNECTIONS: List[Tuple[str, str]] = [
    ("S3", "S1"),
    ("S3", "S2"),
    ("S3", "S4"),
    ("S1", "S2"),
    ("S4", "S5"),
    ("S2", "S5"),
    ("S3", "S5"),
]


class SensorSimulator:
    """Simulates multi-sensor IoT time-series readings with fault injection and neighbor coupling.

    Attributes:
        seed: Random seed for reproducible Gaussian noise.
        baselines: Nominal baseline values for sensors S1 through S5.
        noise_ratio: Relative Gaussian noise standard deviation (~1.5% of baseline).
        coupling_strength: Fraction of anomalous deviation transmitted to neighbors (40%).
        coupling_delay: Time steps delay before neighbor reflects coupled deviation (2 steps).
        connections: Graph edges representing physical coupling paths.
    """

    def __init__(
        self,
        seed: int = 42,
        baselines: Optional[Dict[str, float]] = None,
        noise_ratio: float = 0.015,
        coupling_strength: float = 0.40,
        coupling_delay: int = 2,
        connections: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Initialize simulator with random seed and network configuration.

        Args:
            seed: Integer random seed for reproducible simulations (default: 42).
            baselines: Dict mapping sensor ID to baseline value. Defaults to S1..S5 standard baselines.
            noise_ratio: Standard deviation of Gaussian noise as fraction of baseline (default: 0.015).
            coupling_strength: Fractional influence transferred to connected neighbors (default: 0.40).
            coupling_delay: Time step delay for neighbor coupling propagation (default: 2).
            connections: List of (sensor_a, sensor_b) edge tuples. Defaults to PROJECT_RULES.md topology.
        """
        self.seed = seed
        self.baselines = baselines.copy() if baselines is not None else DEFAULT_BASELINES.copy()
        self.noise_ratio = noise_ratio
        self.coupling_strength = coupling_strength
        self.coupling_delay = coupling_delay
        self.connections = connections if connections is not None else DEFAULT_CONNECTIONS.copy()
        self.sensors = list(self.baselines.keys())

        # Build adjacency lookup dictionary for fast neighbor querying with duplicate protection
        self.adjacency: Dict[str, List[str]] = {s: [] for s in self.sensors}
        for edge in self.connections:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                continue
            u, v = edge[0], edge[1]
            if u == v or u not in self.adjacency or v not in self.adjacency:
                continue
            if v not in self.adjacency[u]:
                self.adjacency[u].append(v)
            if u not in self.adjacency[v]:
                self.adjacency[v].append(u)

    def _get_neighbors(self, sensor: str) -> List[str]:
        """Return list of directly connected neighbor sensor IDs."""
        return self.adjacency.get(sensor, [])

    def generate(
        self,
        n_steps: int = 300,
        fault: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate multi-sensor readings over n_steps with optional fault injection and coupling.

        CONSISTENT DEFINITION OF MAGNITUDE:
        ------------------------------------
        For all fault types, `magnitude` represents the target physical deviation (offset)
        from the sensor's nominal baseline in measurement units:
          - 'drift': The sensor deviates gradually, ramping linearly from 0 at `start`
                     to `+magnitude` at the end of the simulation (or over `duration`).
          - 'spike': An abrupt transient impulse of amplitude `+magnitude` added to the
                     sensor reading starting at `start` for a brief duration (default: 5 steps).
          - 'stuck': The sensor value becomes frozen from `start` step onwards at a constant
                     value equal to `baseline + magnitude` (variance / noise drops to 0).

        COUPLING DYNAMICS:
        ------------------
        When an injected fault creates an anomalous deviation on the target sensor, 40% of
        that deviation is transmitted to its directly connected topological neighbors with
        a 2-step delay. Neighbor deviation is scaled proportionally to preserve relative physical
        coupling across sensors with different units (°C, bar, mm/s, etc.):
            coupled_deviation_neighbor(t) = coupling_strength * (target_dev(t - delay) / target_baseline) * neighbor_baseline

        Args:
            n_steps: Total number of time steps to generate (e.g. 300).
            fault: Optional dictionary specifying fault parameters, e.g.:
                   {"sensor": "S3", "type": "drift", "start": 150, "magnitude": 3.0}
                   Optional keys: 'duration' (for spike/drift duration).

        Returns:
            pandas DataFrame with columns: ['time', 'S1', 'S2', 'S3', 'S4', 'S5']
            and ground-truth fault info attached in df.attrs['fault'].
        """
        rng = np.random.default_rng(self.seed)

        # 1. Generate base normal readings with ~1.5% Gaussian noise for each sensor
        # noise ~ N(0, (baseline * noise_ratio)^2)
        raw_signals: Dict[str, np.ndarray] = {}
        for sensor in self.sensors:
            baseline = self.baselines[sensor]
            noise_std = baseline * self.noise_ratio
            noise = rng.normal(loc=0.0, scale=noise_std, size=n_steps)
            raw_signals[sensor] = baseline + noise

        # 2. Track pure anomalous deviations injected into each sensor (excluding normal noise)
        injected_dev: Dict[str, np.ndarray] = {s: np.zeros(n_steps, dtype=float) for s in self.sensors}

        # Validate and apply injected fault if specified
        if fault is not None:
            target_sensor = fault.get("sensor")
            fault_type = fault.get("type")
            start = int(fault.get("start", 0))
            magnitude = float(fault.get("magnitude", 0.0))
            duration = fault.get("duration", None)

            if target_sensor not in self.sensors:
                raise ValueError(f"Unknown fault target sensor '{target_sensor}'. Must be one of {self.sensors}.")
            if fault_type not in {"drift", "spike", "stuck"}:
                raise ValueError(f"Unsupported fault type '{fault_type}'. Supported: 'drift', 'spike', 'stuck'.")
            if start < 0 or start >= n_steps:
                raise ValueError(f"Fault start step {start} is out of bounds for n_steps={n_steps}.")

            target_baseline = self.baselines[target_sensor]

            if fault_type == "drift":
                # Gradual linear increase or decrease over time.
                # Magnitude Convention:
                # - 'magnitude' specifies the exact target physical deviation (offset) from baseline.
                # - When duration is None, progress ramps from 0.0 at t=start to reach exactly 1.0 (magnitude)
                #   at the final simulation step (t = n_steps - 1).
                # - When duration is explicitly set, progress ramps from 0.0 at t=start to reach exactly 1.0
                #   at t = start + duration - 1, and remains clamped at magnitude thereafter.
                # - For 1-step drift or drift starting at the final timestep, ramp_steps is 0 and
                #   progress is immediately 1.0 (full magnitude).
                if duration is not None:
                    ramp_steps = max(int(duration) - 1, 0)
                else:
                    ramp_steps = max(n_steps - 1 - start, 0)

                for t in range(start, n_steps):
                    if ramp_steps == 0:
                        progress = 1.0
                    else:
                        progress = min((t - start) / ramp_steps, 1.0)
                    dev = magnitude * progress
                    injected_dev[target_sensor][t] = dev
                    raw_signals[target_sensor][t] += dev

            elif fault_type == "spike":
                # Sudden transient spike lasting for duration steps (default: 5 steps)
                spike_len = int(duration) if duration is not None else 5
                end_step = min(start + spike_len, n_steps)
                for t in range(start, end_step):
                    injected_dev[target_sensor][t] = magnitude
                    raw_signals[target_sensor][t] += magnitude

            elif fault_type == "stuck":
                # Sensor freezes at a fixed value: baseline + magnitude (noise vanishes)
                frozen_value = target_baseline + magnitude
                for t in range(start, n_steps):
                    dev = frozen_value - target_baseline
                    injected_dev[target_sensor][t] = dev
                    raw_signals[target_sensor][t] = frozen_value

            # 3. Propagate 40% neighbor coupling with 2-step delay
            # Coupling affects ONLY direct topological neighbors of target_sensor
            neighbors = self._get_neighbors(target_sensor)
            delay = self.coupling_delay
            strength = self.coupling_strength

            for neighbor in neighbors:
                neighbor_baseline = self.baselines[neighbor]
                for t in range(start + delay, n_steps):
                    # Source deviation at (t - delay)
                    src_dev = injected_dev[target_sensor][t - delay]
                    # Relative deviation on source transferred to neighbor
                    rel_dev = src_dev / target_baseline
                    coupled_effect = strength * rel_dev * neighbor_baseline
                    raw_signals[neighbor][t] += coupled_effect

        # 4. Construct output DataFrame with required schema
        data = {"time": np.arange(n_steps, dtype=int)}
        for sensor in self.sensors:
            data[sensor] = raw_signals[sensor]

        df = pd.DataFrame(data)

        # Store ground-truth fault information in df.attrs['fault']
        if fault is not None:
            df.attrs["fault"] = dict(fault)
        else:
            df.attrs["fault"] = None

        return df

    # Helper alias for backward compatibility with pipeline and tests
    def generate_normal_stream(self, num_samples: int = 100) -> pd.DataFrame:
        """Alias generating clean normal readings over num_samples."""
        df = self.generate(n_steps=num_samples, fault=None)
        df["timestep"] = df["time"]
        return df

    def simulate_network(
        self,
        total_steps: int = 200,
        fault_sensor: Optional[str] = None,
        fault_type: Optional[str] = None,
        fault_start: int = 100,
        fault_duration: int = 50,
        magnitude: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compatibility wrapper connecting legacy parameters to generate()."""
        fault = None
        if fault_sensor and fault_type:
            mag = magnitude if magnitude is not None else (self.baselines[fault_sensor] * 0.5)
            fault = {
                "sensor": fault_sensor,
                "type": fault_type,
                "start": fault_start,
                "magnitude": mag,
                "duration": fault_duration,
            }
        return self.generate(n_steps=total_steps, fault=fault)
