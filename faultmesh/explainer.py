"""Explainable Fault Investigation Engine for FaultMesh AI 2.0.

Responsibility:
    Decomposes and explains root-cause localization decisions using actual runtime
    telemetry, detection onsets, anomaly scores, and sensor network topology.
    Produces structured evidence chains, timing comparisons, neighbor propagation
    analysis, transparent confidence categories (Strong, Moderate, Limited, Insufficient),
    and plain-text investigation reports without treating ranking scores as calibrated
    probabilities.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from faultmesh.graph import SensorNetworkGraph


@dataclass
class InvestigationReport:
    """Structured report produced by the Explainable Fault Investigation Engine."""
    suspected_sensor: Optional[str]
    localization_rank: Optional[int]
    localization_score: Optional[float]
    anomaly_score: Optional[float]
    detection_onset: Optional[int]
    supporting_sensors: List[str]
    evidence: List[str]
    confidence_category: str  # 'Strong', 'Moderate', 'Limited', 'Insufficient'
    investigation_summary: str
    limitations: List[str]
    timing_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    neighbor_evidence: List[Dict[str, Any]] = field(default_factory=list)
    ranking_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence_level(self) -> str:
        """Alias for confidence_category for backward compatibility."""
        return self.confidence_category

    @property
    def primary_suspect(self) -> Optional[str]:
        """Alias for suspected_sensor for backward compatibility."""
        return self.suspected_sensor

    @property
    def summary_narrative(self) -> str:
        """Alias for investigation_summary for backward compatibility."""
        return self.investigation_summary


DEFAULT_LIMITATIONS: List[str] = [
    "Simulated IoT telemetry: Data is synthesized via physics-informed equations, not physical plant sensors.",
    "Heuristic ranking: Localization scores represent multi-criteria fusion weights, not calibrated causal probabilities.",
    "Observational inference: Causal direction is inferred from temporal precedence and topological adjacency without physical intervention.",
    "Hypothesis status: Suspected root-cause sensor represents an algorithmic diagnosis requiring physical technician verification.",
]


class FaultExplainer:
    """Modular Explainable Fault Investigation Engine for FaultMesh AI."""

    def __init__(self, network_graph: Optional[SensorNetworkGraph] = None) -> None:
        """Initialize explainer with sensor network topology.

        Args:
            network_graph: SensorNetworkGraph instance (defaults to standard mesh if None).
        """
        self.graph = network_graph or SensorNetworkGraph()

    def explain(
        self,
        scores_df: Optional[pd.DataFrame] = None,
        onsets: Optional[Dict[str, Optional[int]]] = None,
        rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
        anomaly_scores: Optional[Dict[str, float]] = None,
        diagnosis_status: Optional[str] = None,
        primary_suspect: Optional[str] = None,
    ) -> InvestigationReport:
        """Generate explainable investigation report from actual pipeline results.

        Args:
            scores_df: Optional DataFrame of calibrated anomaly scores over time.
            onsets: Optional Dict mapping sensor ID to first alarm onset timestep.
            rankings: Optional sorted list of (sensor_id, composite_score, breakdown_dict).
            anomaly_scores: Optional Dict mapping sensor ID to scalar anomaly score.
            diagnosis_status: Optional status ("NORMAL", "UNCERTAIN", "CONFIRMED").
            primary_suspect: Optional verified primary suspect sensor ID.

        Returns:
            InvestigationReport containing structured evidence, confidence, and narrative.
        """
        onsets_clean: Dict[str, Optional[int]] = onsets.copy() if onsets is not None else {}
        nodes: List[str] = list(self.graph.nodes)

        # Normalize scalar anomaly scores
        resolved_anomaly_scores: Dict[str, float] = {}
        if anomaly_scores:
            resolved_anomaly_scores = {s: float(v) for s, v in anomaly_scores.items()}
        elif scores_df is not None and not scores_df.empty:
            for s in nodes:
                if s in scores_df.columns:
                    resolved_anomaly_scores[s] = float(scores_df[s].max())
                else:
                    resolved_anomaly_scores[s] = 0.0

        # Handle explicit NORMAL status
        if diagnosis_status == "NORMAL":
            return self._handle_normal_scenario(nodes, resolved_anomaly_scores)

        # Handle empty/missing rankings scenario
        if not rankings:
            return self._handle_unranked_scenario(resolved_anomaly_scores, onsets_clean, nodes)

        # Handle explicit UNCERTAIN status
        if diagnosis_status == "UNCERTAIN":
            return self._handle_uncertain_scenario(nodes, resolved_anomaly_scores, onsets_clean, rankings)

        # Retrieve highest-ranked suspect
        suspect_sensor = primary_suspect or rankings[0][0]
        localization_rank = 1
        localization_score = float(rankings[0][1])
        breakdown = rankings[0][2] if len(rankings[0]) > 2 else {}

        suspect_onset = onsets_clean.get(suspect_sensor)
        suspect_anomaly_score = resolved_anomaly_scores.get(
            suspect_sensor,
            float(breakdown.get("own_anomaly", 0.0)),
        )

        # Identify all alarmed sensors across network
        alarmed_sensors = {s: t for s, t in onsets_clean.items() if t is not None}

        # Check if entire network remained normal
        if not alarmed_sensors and (suspect_anomaly_score < 0.50 and localization_score < 0.40):
            return self._handle_normal_scenario(nodes, resolved_anomaly_scores)

        # 1. Timing Precedence & Sensor Comparisons
        timing_comparisons: List[Dict[str, Any]] = []
        earliest_step: Optional[int] = min(alarmed_sensors.values()) if alarmed_sensors else None
        earliest_sensors: List[str] = [s for s, t in alarmed_sensors.items() if t == earliest_step]

        for s in nodes:
            t_s = onsets_clean.get(s)
            score_s = resolved_anomaly_scores.get(s, 0.0)

            if t_s is None:
                timing_status = "Nominal (No alarm triggered)"
                delay_str = "None"
            elif suspect_onset is not None:
                if s == suspect_sensor:
                    timing_status = "Suspected root cause"
                    delay_str = "0 (Reference)"
                elif t_s > suspect_onset:
                    delay_steps = t_s - suspect_onset
                    timing_status = f"Delayed cascade (+{delay_steps} steps after {suspect_sensor})"
                    delay_str = f"+{delay_steps}"
                elif t_s < suspect_onset:
                    lead_steps = suspect_onset - t_s
                    timing_status = f"Preceded suspect (-{lead_steps} steps earlier)"
                    delay_str = f"-{lead_steps}"
                else:
                    timing_status = f"Concurrent alarm (t={t_s})"
                    delay_str = "0 (Concurrent)"
            else:
                timing_status = f"Alarm at t={t_s}"
                delay_str = "N/A"

            timing_comparisons.append({
                "sensor": s,
                "onset": t_s,
                "timing_status": timing_status,
                "delay_relative_to_suspect": delay_str,
                "anomaly_score": round(score_s, 4),
            })

        # 2. Neighbor Propagation Support
        neighbors = self.graph.get_neighbors(suspect_sensor)
        neighbor_evidence: List[Dict[str, Any]] = []
        supporting_sensors: List[str] = []

        for nb in neighbors:
            t_nb = onsets_clean.get(nb)
            nb_score = resolved_anomaly_scores.get(nb, 0.0)

            if t_nb is not None and suspect_onset is not None and t_nb >= suspect_onset:
                delay = t_nb - suspect_onset
                relation = f"Downstream propagation (+{delay} steps delayed)"
                is_support = True
                supporting_sensors.append(nb)
            elif t_nb is not None and suspect_onset is not None and t_nb < suspect_onset:
                lead = suspect_onset - t_nb
                relation = f"Upstream trigger ({lead} steps before suspect)"
                is_support = False
            elif t_nb is not None:
                relation = f"Alarm at t={t_nb}"
                is_support = False
            else:
                relation = "Unaffected nominal neighbor"
                is_support = False

            neighbor_evidence.append({
                "neighbor": nb,
                "onset": t_nb,
                "anomaly_score": round(nb_score, 4),
                "relationship": relation,
                "supports_suspect": is_support,
            })

        # 3. Structured Evidence Points
        evidence: List[str] = []

        # Point A: Earliest onset identification
        if suspect_onset is not None:
            if suspect_onset == earliest_step:
                if len(earliest_sensors) == 1:
                    evidence.append(
                        f"Sensor {suspect_sensor} first exhibited sustained abnormal behavior at time step {suspect_onset}, strictly leading all other sensors."
                    )
                else:
                    evidence.append(
                        f"Sensor {suspect_sensor} alarmed at time step {suspect_onset}, tied for earliest onset with {', '.join(s for s in earliest_sensors if s != suspect_sensor)}."
                    )
            else:
                delay_from_earliest = suspect_onset - earliest_step if earliest_step is not None else 0
                evidence.append(
                    f"Sensor {suspect_sensor} showed abnormal behavior at time step {suspect_onset}, which occurred {delay_from_earliest} steps after initial alarms on {', '.join(earliest_sensors)}."
                )
        else:
            evidence.append(
                f"Sensor {suspect_sensor} did not manifest a sustained alarm onset exceeding the confirmation threshold."
            )

        # Point B: Subsequent abnormal sensors
        subsequent_sensors = [
            f"{s} (t={t})" for s, t in alarmed_sensors.items()
            if suspect_onset is not None and t > suspect_onset
        ]
        if subsequent_sensors:
            evidence.append(
                f"Sensors showing subsequent abnormal behavior afterward: {', '.join(subsequent_sensors)}."
            )
        elif suspect_onset is not None and len(alarmed_sensors) == 1:
            evidence.append("No other sensors triggered sustained alarms; the anomaly remained localized strictly to the candidate.")
        else:
            evidence.append("Multiple sensors alarmed simultaneously without confirmed subsequent delay cascades.")

        # Point C: Neighboring sensor support
        if supporting_sensors:
            evidence.append(
                f"Connected topological neighbor(s) {', '.join(supporting_sensors)} support {suspect_sensor} as the originating source through delayed abnormal propagation."
            )
        else:
            evidence.append(
                f"Directly connected neighbors ({', '.join(neighbors)}) did not manifest delayed propagation supporting {suspect_sensor}."
            )

        # Point D: Anomaly score comparison
        runner_up = rankings[1] if len(rankings) > 1 else None
        runner_up_str = f"compared to runner-up {runner_up[0]} (anomaly: {resolved_anomaly_scores.get(runner_up[0], 0.0):.3f})" if runner_up else ""
        evidence.append(
            f"Sensor {suspect_sensor} exhibited a peak anomaly score of {suspect_anomaly_score:.3f} {runner_up_str}."
        )

        # Point E: Localization ranking support
        evidence.append(
            f"Graph-guided multi-criteria localization ranked {suspect_sensor} #1 with composite score {localization_score:.4f} "
            f"(own anomaly: {breakdown.get('own_anomaly', 0.0):.3f}, earliness: {breakdown.get('detection_earliness', 0.0):.3f}, "
            f"neighbor support: {breakdown.get('neighbour_support', 0.0):.3f}, coverage: {breakdown.get('coverage', 0.0):.3f})."
        )

        # 4. Confidence Categorization (Strong, Moderate, Limited, Insufficient)
        score_margin = (localization_score - runner_up[1]) if runner_up else 1.0

        if not alarmed_sensors or suspect_onset is None:
            confidence_category = "Insufficient"
            confidence_reason = "No confirmed alarm onset declared for the suspect sensor."
        elif (
            suspect_onset == earliest_step
            and len(earliest_sensors) == 1
            and suspect_anomaly_score >= 0.70
            and len(supporting_sensors) >= 1
            and score_margin >= 0.04
        ):
            confidence_category = "Strong"
            confidence_reason = (
                f"Clear earliest onset (t={suspect_onset}), robust anomaly intensity ({suspect_anomaly_score:.2f}), "
                f"confirmed propagation to {len(supporting_sensors)} connected neighbor(s), and decisive score separation (+{score_margin:.3f})."
            )
        elif (
            suspect_onset is not None
            and (earliest_step is not None and suspect_onset - earliest_step <= 2)
            and suspect_anomaly_score >= 0.50
        ):
            confidence_category = "Moderate"
            if len(supporting_sensors) > 0:
                confidence_reason = (
                    f"Candidate alarmed early (t={suspect_onset}) with moderate anomaly intensity ({suspect_anomaly_score:.2f}) "
                    f"and neighbor support, though timing or ranking margin is narrow."
                )
            else:
                confidence_reason = (
                    f"Candidate alarmed earliest (t={suspect_onset}) with elevated anomaly score ({suspect_anomaly_score:.2f}), "
                    f"but connected neighbors showed minimal delayed cascade."
                )
        else:
            confidence_category = "Limited"
            confidence_reason = (
                f"Candidate did not strictly lead timing or exhibits low anomaly intensity ({suspect_anomaly_score:.2f}); "
                f"suspect may be an alarmed coupled recipient rather than true root cause."
            )

        # 5. Formulate Investigation Summary Report
        summary_lines = [
            "FAULT INVESTIGATION REPORT",
            "==========================",
            f"Suspected Sensor: {suspect_sensor}",
            f"Localization Rank: #{localization_rank} (Score: {localization_score:.4f})",
            f"Detection Onset: {'t = ' + str(suspect_onset) if suspect_onset is not None else 'None'}",
            f"Peak Anomaly Score: {suspect_anomaly_score:.3f}",
            "",
            "Observed Evidence:",
        ]
        for item in evidence:
            summary_lines.append(f"- {item}")

        summary_lines.extend([
            "",
            "Investigation Conclusion:",
            f"{suspect_sensor} is the highest-ranked suspected fault source based on the available simulated telemetry and topological propagation evidence.",
            "",
            "Confidence Category:",
            f"Evidence Strength: {confidence_category} ({confidence_reason})",
            "",
            "Limitations:",
        ])
        for lim in DEFAULT_LIMITATIONS:
            summary_lines.append(f"- {lim}")

        investigation_summary = "\n".join(summary_lines)

        return InvestigationReport(
            suspected_sensor=suspect_sensor,
            localization_rank=localization_rank,
            localization_score=round(localization_score, 4),
            anomaly_score=round(suspect_anomaly_score, 4),
            detection_onset=suspect_onset,
            supporting_sensors=supporting_sensors,
            evidence=evidence,
            confidence_category=confidence_category,
            investigation_summary=investigation_summary,
            limitations=DEFAULT_LIMITATIONS.copy(),
            timing_comparisons=timing_comparisons,
            neighbor_evidence=neighbor_evidence,
            ranking_breakdown={
                k: round(float(v), 4) for k, v in breakdown.items() if isinstance(v, (int, float))
            },
        )

    def _handle_uncertain_scenario(
        self,
        nodes: List[str],
        anomaly_scores: Dict[str, float],
        onsets: Dict[str, Optional[int]],
        rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
    ) -> InvestigationReport:
        """Handle scenarios where anomalous behavior was observed but root cause cannot be confirmed."""
        max_score = max(anomaly_scores.values()) if anomaly_scores else 0.0
        elevated_sensors = [s for s, sc in anomaly_scores.items() if sc >= 0.50]
        alarmed_sensors = [s for s, t in onsets.items() if t is not None]

        evidence = [
            f"Sensor network exhibited elevated anomaly scores (peak: {max_score:.3f}).",
            "Root-cause attribution is uncertain: lack of distinct chronological propagation delay or persistent onset confirmation.",
        ]
        if alarmed_sensors:
            evidence.append(f"Sensors alarming concurrently without decisive cascade: {', '.join(alarmed_sensors)}.")
        elif elevated_sensors:
            evidence.append(f"Sensors exhibiting transient or unconfirmed elevated readings: {', '.join(elevated_sensors)}.")
        else:
            evidence.append("No sustained onset was confirmed exceeding the 3-consecutive-reading threshold (0.60).")

        top_score = float(rankings[0][1]) if rankings else 0.0
        summary_text = (
            "FAULT INVESTIGATION REPORT\n"
            "==========================\n"
            "Suspected Root Cause: Root Cause Uncertain\n"
            "Localization Rank: N/A (Inconclusive)\n"
            "Detection Onset: None\n"
            "Peak Anomaly Score: " + f"{max_score:.3f}\n\n"
            "Observed Evidence:\n"
            + "\n".join(f"- {ev}" for ev in evidence)
            + "\n\nInvestigation Conclusion:\n"
            "Elevated sensor deviations observed across the network mesh, but available causal propagation evidence is insufficient "
            "to reliably isolate a single primary root cause. Root cause is classified as UNCERTAIN.\n\n"
            "Confidence Category:\n"
            "Evidence Strength: Limited (Uncertain root-cause attribution due to concurrent alarms or transient noise)\n\n"
            "Limitations:\n"
            "- Multi-sensor ambiguity: When multiple nodes alarm simultaneously, directional causality requires secondary physical inspection."
        )

        timing_comps = [
            {
                "sensor": s,
                "onset": onsets.get(s),
                "timing_status": f"Alarm at t={onsets.get(s)}" if onsets.get(s) is not None else "Nominal / Transient",
                "delay_relative_to_suspect": "N/A",
                "anomaly_score": round(anomaly_scores.get(s, 0.0), 4),
            }
            for s in nodes
        ]

        return InvestigationReport(
            suspected_sensor=None,
            localization_rank=None,
            localization_score=round(top_score, 4),
            anomaly_score=round(max_score, 4),
            detection_onset=None,
            supporting_sensors=[],
            evidence=evidence,
            confidence_category="Limited",
            investigation_summary=summary_text,
            limitations=DEFAULT_LIMITATIONS.copy(),
            timing_comparisons=timing_comps,
            neighbor_evidence=[],
            ranking_breakdown={},
        )

    def _handle_normal_scenario(
        self,
        nodes: List[str],
        anomaly_scores: Dict[str, float],
    ) -> InvestigationReport:
        """Handle scenarios where all sensors operated normally with no detected faults."""
        max_score = max(anomaly_scores.values()) if anomaly_scores else 0.0
        evidence = [
            "All sensor anomaly scores remained below the configured alarm threshold (0.60).",
            "No sustained alarm onsets were confirmed across the sensor mesh network.",
            "All multi-sensor readings remained within nominal baseline operating boundaries.",
        ]

        summary_text = (
            "FAULT INVESTIGATION REPORT\n"
            "==========================\n"
            "Suspected Sensor: None\n"
            "Localization Rank: N/A\n"
            "Detection Onset: None\n"
            "Peak Anomaly Score: " + f"{max_score:.3f}\n\n"
            "Observed Evidence:\n"
            "- All sensor anomaly scores remained below the configured alarm threshold (0.60).\n"
            "- No sustained alarm onsets were confirmed across the sensor mesh network.\n"
            "- All multi-sensor readings remained within nominal baseline operating boundaries.\n\n"
            "Investigation Conclusion:\n"
            "All sensors operated within normal nominal boundaries. No abnormal fault onset was detected.\n\n"
            "Confidence Category:\n"
            "Evidence Strength: Insufficient (Clean nominal operation without abnormal behavior)\n\n"
            "Limitations:\n"
            "- Simulated IoT telemetry: Data is synthesized via physics-informed equations, not physical plant sensors."
        )

        return InvestigationReport(
            suspected_sensor=None,
            localization_rank=None,
            localization_score=0.0,
            anomaly_score=round(max_score, 4),
            detection_onset=None,
            supporting_sensors=[],
            evidence=evidence,
            confidence_category="Insufficient",
            investigation_summary=summary_text,
            limitations=DEFAULT_LIMITATIONS.copy(),
            timing_comparisons=[
                {"sensor": s, "onset": None, "timing_status": "Nominal", "delay_relative_to_suspect": "N/A", "anomaly_score": round(anomaly_scores.get(s, 0.0), 4)}
                for s in nodes
            ],
            neighbor_evidence=[],
            ranking_breakdown={},
        )

    def _handle_unranked_scenario(
        self,
        anomaly_scores: Dict[str, float],
        onsets: Dict[str, Optional[int]],
        nodes: List[str],
    ) -> InvestigationReport:
        """Handle scenarios where rankings are absent or empty."""
        evidence = ["No localization ranking results were provided for investigation."]
        summary_text = (
            "FAULT INVESTIGATION REPORT\n"
            "==========================\n"
            "Suspected Sensor: None\n"
            "Investigation Conclusion: Unable to conduct fault investigation due to missing sensor rankings.\n"
            "Confidence Category: Evidence Strength: Insufficient (Missing ranking inputs)"
        )

        return InvestigationReport(
            suspected_sensor=None,
            localization_rank=None,
            localization_score=0.0,
            anomaly_score=0.0,
            detection_onset=None,
            supporting_sensors=[],
            evidence=evidence,
            confidence_category="Insufficient",
            investigation_summary=summary_text,
            limitations=DEFAULT_LIMITATIONS.copy(),
            timing_comparisons=[],
            neighbor_evidence=[],
            ranking_breakdown={},
        )
