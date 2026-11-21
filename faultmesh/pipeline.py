"""Pipeline module for FaultMesh AI.

Responsibility:
    Connect and orchestrate simulation, feature processing, anomaly detection,
    and graph-guided fault localization into a unified workflow.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from faultmesh.simulator import SensorSimulator
from faultmesh.processing import FeatureProcessor
from faultmesh.detector import AnomalyDetector
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer


class FaultMeshPipeline:
    """Orchestrates the complete FaultMesh AI end-to-end processing pipeline."""

    def __init__(
        self,
        simulator: Optional[SensorSimulator] = None,
        processor: Optional[FeatureProcessor] = None,
        detector: Optional[AnomalyDetector] = None,
        graph: Optional[SensorNetworkGraph] = None,
        localizer: Optional[FaultLocalizer] = None,
    ) -> None:
        """Initialize the orchestration pipeline with sub-components.

        Args:
            simulator: SensorSimulator instance.
            processor: FeatureProcessor instance.
            detector: AnomalyDetector instance.
            graph: SensorNetworkGraph instance.
            localizer: FaultLocalizer instance.
        """
        self.simulator = simulator or SensorSimulator()
        self.processor = processor or FeatureProcessor()
        self.detector = detector or AnomalyDetector()
        self.graph = graph or SensorNetworkGraph()
        self.localizer = localizer or FaultLocalizer(network_graph=self.graph)

    def run_pipeline(
        self,
        fault_sensor: Optional[str] = "S1",
        fault_type: Optional[str] = "drift",
        total_steps: int = 200,
        fault_start: int = 100,
        fault_duration: int = 50,
        baseline_samples: int = 100,
    ) -> Dict[str, Any]:
        """Execute end-to-end simulation, feature extraction, detection, and localization.

        Args:
            fault_sensor: Sensor to inject fault into (or None for baseline-only run).
            fault_type: Type of fault ('drift', 'spike', 'stuck').
            total_steps: Total duration of simulation run.
            fault_start: Timestep when fault is triggered.
            fault_duration: Number of steps fault persists.
            baseline_samples: Number of initial samples used for detector training.

        Returns:
            Dictionary containing:
                - 'raw_readings': DataFrame of generated sensor signals
                - 'features': DataFrame of extracted statistical features
                - 'detections': DataFrame with anomaly scores and alarm flags
                - 'alarm_steps': Dict of first alarm timestep per sensor
                - 'rankings': Ranked list of suspected faulty sensors
        """
        # TODO: Implement step-by-step pipeline execution
        # 1. Simulate readings
        raw_df = self.simulator.simulate_network(
            total_steps=total_steps,
            fault_sensor=fault_sensor,
            fault_type=fault_type,
            fault_start=fault_start,
            fault_duration=fault_duration,
        )

        sensor_cols = self.graph.nodes

        # 2. Extract features
        self.processor.fit_baseline(raw_df.iloc[:baseline_samples], sensor_cols)
        features_df = self.processor.extract_features(raw_df, sensor_cols)

        # 3. Fit detectors on baseline and evaluate
        self.detector.fit_all(features_df.iloc[:baseline_samples], sensor_cols)
        detections_df, alarm_steps = self.detector.evaluate_detections(features_df, sensor_cols)

        # 4. Localize root cause
        peak_scores = {
            s: float(detections_df[f"{s}_anomaly_score"].max())
            for s in sensor_cols
            if f"{s}_anomaly_score" in detections_df.columns
        }
        rankings = self.localizer.rank_suspects(peak_scores, alarm_steps, total_steps)

        return {
            "raw_readings": raw_df,
            "features": features_df,
            "detections": detections_df,
            "alarm_steps": alarm_steps,
            "rankings": rankings,
        }
