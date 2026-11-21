"""Fault localization module for FaultMesh AI.

Responsibility:
    Rank suspected faulty sensors based on multi-criteria cooperative scoring:
    own anomaly score (35%), detection earliness (30%), neighbor support (20%),
    and graph coverage (15%).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from faultmesh.graph import SensorNetworkGraph


# Localization weights according to PROJECT_RULES.md
DEFAULT_WEIGHTS: Dict[str, float] = {
    "own_anomaly": 0.35,          # 35%
    "detection_earliness": 0.30,  # 30%
    "neighbour_support": 0.20,    # 20%
    "coverage": 0.15,             # 15%
}


class FaultLocalizer:
    """Ranks candidate root-cause faulty sensors using graph-guided multi-criteria fusion."""

    def __init__(
        self,
        network_graph: Optional[SensorNetworkGraph] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize fault localizer with sensor network graph and criterion weights.

        Args:
            network_graph: Instance of SensorNetworkGraph (creates default if None).
            weights: Dictionary with keys 'own_anomaly', 'detection_earliness',
                     'neighbour_support', 'coverage'.
        """
        self.graph = network_graph or SensorNetworkGraph()
        self.weights = weights.copy() if weights is not None else DEFAULT_WEIGHTS.copy()

    def compute_earliness_score(
        self,
        first_alarm_steps: Dict[str, Optional[int]],
        max_timesteps: int = 200,
    ) -> Dict[str, float]:
        """Compute normalized detection earliness score for each sensor.

        MATHEMATICAL FORMULA:
        ---------------------
        For sensors with valid detection onsets:
            earliness(sensor) = (max_onset - sensor_onset) / (max_onset - min_onset)

        Edge Cases:
        - If no sensors triggered alarms: all sensors receive 0.0.
        - If all valid onsets are identical (max == min): all alarmed sensors receive 1.0.
        - Sensors without a valid detection onset (onset is None): receive 0.0.

        Args:
            first_alarm_steps: Dict mapping sensor ID to first alarm timestep or None.
            max_timesteps: Upper bound on total simulation steps (unused in relative span).

        Returns:
            Dict mapping sensor ID to earliness score strictly in [0.0, 1.0].
        """
        earliness: Dict[str, float] = {node: 0.0 for node in self.graph.nodes}
        valid_steps = [step for step in first_alarm_steps.values() if step is not None]

        if not valid_steps:
            return earliness

        min_step = min(valid_steps)
        max_step = max(valid_steps)
        span = max_step - min_step

        for node in self.graph.nodes:
            step = first_alarm_steps.get(node)
            if step is None:
                earliness[node] = 0.0
            elif span == 0:
                earliness[node] = 1.0
            else:
                earliness[node] = float((max_step - step) / span)

        return earliness

    def compute_neighbour_support(
        self,
        anomaly_scores: Union[Dict[str, float], List[str]],
        first_alarm_steps: Optional[Dict[str, Optional[int]]] = None,
    ) -> Dict[str, float]:
        """Evaluate whether connected neighbors exhibit abnormal behavior consistent with delayed propagation.

        MATHEMATICAL FORMULA:
        ---------------------
        A candidate sensor u is supported as root cause if its connected neighbors v:
            1. Manifested an alarm at or after u's detection onset (t_v >= t_u).
            2. Present strong anomaly score evidence.

        For sensor u:
            support(u) = sum(anomaly_scores[v] for v in neighbors(u) if t_v >= t_u) / |neighbors(u)|

        If sensor u never triggered an alarm (t_u is None), it receives support(u) = 0.0.

        Args:
            anomaly_scores: Dict mapping sensor ID to anomaly score (or legacy list of alarmed sensors).
            first_alarm_steps: Dict mapping sensor ID to first alarm timestep.

        Returns:
            Dict mapping sensor ID to neighbor support score in [0.0, 1.0].
        """
        # Backward compatibility for legacy test calls passing only a list of alarmed sensor names
        if isinstance(anomaly_scores, (list, set)) and first_alarm_steps is None:
            alarm_set = set(anomaly_scores)
            support: Dict[str, float] = {}
            for node in self.graph.nodes:
                nbs = self.graph.get_neighbors(node)
                if not nbs:
                    support[node] = 0.0
                else:
                    alarmed_nbs = [nb for nb in nbs if nb in alarm_set]
                    support[node] = float(len(alarmed_nbs) / len(nbs))
            return support

        scores_dict: Dict[str, float] = anomaly_scores if isinstance(anomaly_scores, dict) else {}
        onsets_dict: Dict[str, Optional[int]] = first_alarm_steps or {}
        support = {}

        for node in self.graph.nodes:
            t_node = onsets_dict.get(node)
            neighbors = self.graph.get_neighbors(node)

            if t_node is None or not neighbors:
                # Sensor never alarmed or has no neighbors -> cannot be causal source of neighbor faults
                support[node] = 0.0
            else:
                # Accumulate evidence from neighbors that alarmed at or after candidate node
                causal_evidence = [
                    scores_dict.get(nb, 0.0)
                    for nb in neighbors
                    if onsets_dict.get(nb) is not None and onsets_dict[nb] >= t_node
                ]
                support[node] = float(sum(causal_evidence) / len(neighbors))

        return support

    def compute_coverage_scores(
        self,
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Measure abnormality evidence across the candidate sensor's connected neighborhood.

        MATHEMATICAL FORMULA:
        ---------------------
        Rather than independently rewarding a sensor simply because it has many topological neighbors,
        graph coverage evaluates the local neighborhood abnormality density:
            coverage(u) = sum(anomaly_scores[v] for v in neighbors(u)) / |neighbors(u)|

        This strictly normalizes between 0 and 1 and reflects the spread of disturbance around u.
        If anomaly_scores is None, delegates to normalized graph degree centrality for compatibility.

        Args:
            anomaly_scores: Optional Dict mapping sensor ID to anomaly score.

        Returns:
            Dict mapping sensor ID to coverage score in [0.0, 1.0].
        """
        if anomaly_scores is None:
            return {node: self.graph.get_coverage_score(node) for node in self.graph.nodes}

        coverage: Dict[str, float] = {}
        for node in self.graph.nodes:
            neighbors = self.graph.get_neighbors(node)
            if not neighbors:
                coverage[node] = 0.0
            else:
                neighbor_abnormality = [anomaly_scores.get(nb, 0.0) for nb in neighbors]
                coverage[node] = float(sum(neighbor_abnormality) / len(neighbors))

        return coverage

    def generate_explanation(
        self,
        sensor: str,
        composite: float,
        breakdown: Dict[str, float],
        onset: Optional[int],
        neighbors: List[str],
        onsets: Dict[str, Optional[int]],
    ) -> str:
        """Generate clear diagnostic explanation for a sensor's root-cause ranking."""
        own = breakdown["own_anomaly"]
        early = breakdown["detection_earliness"]
        supp = breakdown["neighbour_support"]
        cov = breakdown["coverage"]

        if onset is None:
            return f"Sensor {sensor} remained normal with no confirmed alarm onset (Score: {composite:.4f})."

        # Check neighbor propagation details
        delayed_neighbors = [nb for nb in neighbors if onsets.get(nb) is not None and onsets[nb] >= onset]
        premature_neighbors = [nb for nb in neighbors if onsets.get(nb) is not None and onsets[nb] < onset]

        if early >= 0.90 and len(delayed_neighbors) > 0:
            return (
                f"Sensor {sensor} is the primary root cause: Alarm triggered earliest at t={onset}; "
                f"{len(delayed_neighbors)}/{len(neighbors)} neighbors manifested delayed anomalous propagation."
            )
        elif len(premature_neighbors) > 0:
            earlier_str = ", ".join(f"{nb} (t={onsets[nb]})" for nb in premature_neighbors)
            return (
                f"Sensor {sensor} is a coupled recipient: Alarm triggered at t={onset}, after {earlier_str}. "
                f"Fault propagated into this sensor from adjacent upstream neighbors."
            )
        else:
            return (
                f"Sensor {sensor} (Score: {composite:.4f}): Onset at t={onset}, own anomaly {own:.3f}, "
                f"neighbor support {supp:.3f}, coverage {cov:.3f}."
            )

    def rank_suspects(
        self,
        anomaly_scores: Dict[str, float],
        first_alarm_steps: Dict[str, Optional[int]],
        total_steps: int = 200,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rank suspected faulty sensors in descending order of composite root-cause score.

        WEIGHTED SCORING:
            composite = 0.35 * own + 0.30 * early + 0.20 * supp + 0.15 * cov

        Args:
            anomaly_scores: Anomaly scores per sensor in [0, 1].
            first_alarm_steps: Dict mapping sensor ID to timestep of first alarm.
            total_steps: Total duration of simulation window.

        Returns:
            Sorted list of tuples: (sensor_id, composite_score, breakdown_dict)
        """
        earliness = self.compute_earliness_score(first_alarm_steps, total_steps)
        support = self.compute_neighbour_support(anomaly_scores, first_alarm_steps)
        coverage = self.compute_coverage_scores(anomaly_scores)

        rankings: List[Tuple[str, float, Dict[str, Any]]] = []

        for sensor in self.graph.nodes:
            own = float(np.clip(anomaly_scores.get(sensor, 0.0), 0.0, 1.0))
            early = float(np.clip(earliness.get(sensor, 0.0), 0.0, 1.0))
            supp = float(np.clip(support.get(sensor, 0.0), 0.0, 1.0))
            cov = float(np.clip(coverage.get(sensor, 0.0), 0.0, 1.0))

            composite = (
                self.weights["own_anomaly"] * own
                + self.weights["detection_earliness"] * early
                + self.weights["neighbour_support"] * supp
                + self.weights["coverage"] * cov
            )
            composite = float(np.clip(composite, 0.0, 1.0))

            neighbors = self.graph.get_neighbors(sensor)
            onset = first_alarm_steps.get(sensor)

            breakdown = {
                "own_anomaly": own,
                "detection_earliness": early,
                "neighbour_support": supp,
                "coverage": cov,
                "onset": onset,
            }

            breakdown["explanation"] = self.generate_explanation(
                sensor=sensor,
                composite=composite,
                breakdown=breakdown,
                onset=onset,
                neighbors=neighbors,
                onsets=first_alarm_steps,
            )

            rankings.append((sensor, composite, breakdown))

        # Sort descending by composite score, tie-break by earliness descending, then sensor_id ascending
        rankings.sort(key=lambda item: (-item[1], -item[2]["detection_earliness"], item[0]))
        return rankings
