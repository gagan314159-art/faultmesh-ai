"""Predictive Maintenance Recommendation Engine for FaultMesh AI 2.0.

Responsibility:
    Generates explainable, evidence-based, physics-informed maintenance recommendations
    for industrial sensors (S1-S5) and rotating machinery. Synthesizes runtime outputs
    from sensor simulation, anomaly detection, root-cause localization, Phase 1
    Explainable Fault Investigation, and Phase 2 Sensor Health Monitoring.
    Classifies maintenance into four operational priority levels:
    - Critical (immediate intervention)
    - High (urgent inspection / calibration)
    - Medium (increased monitoring / routine check)
    - Low (nominal routine maintenance)
    - Insufficient Data (telemetry diagnostics required)
    
    Provides targeted technician checklists and explicit engineering disclaimers.
    Does NOT predict physical equipment failure dates or act as a certified safety controller.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from faultmesh.health import SensorHealthRecord, SensorHealthReport

MAINTENANCE_DISCLAIMER: str = (
    "NOTICE: Maintenance recommendations are advisory, rule-guided physical decision-support "
    "protocols derived from telemetry anomaly patterns, topological graph localization, and sensor "
    "domain knowledge. They provide diagnostic guidance and technician inspection checklists; "
    "they do NOT constitute empirical Remaining Useful Life (RUL) run-to-failure predictions, "
    "failure date forecasts, or certified automatic equipment-control actions."
)

DISCLAIMER_TEXT = MAINTENANCE_DISCLAIMER


@dataclass
class MaintenanceAction:
    """Actionable physical maintenance item."""
    action_id: str
    title: str
    description: str
    component: str
    required_tools: List[str]
    safety_notes: str


class MaintenanceAdvisory:
    """Comprehensive maintenance advisory report for an individual sensor channel."""

    def __init__(
        self,
        sensor_id: str,
        sensor_name: str,
        priority: str,
        recommended_action: str,
        reason: str,
        supporting_evidence: Optional[List[str]] = None,
        health_metrics: Optional[Dict[str, Any]] = None,
        confidence_category: str = "Moderate",
        suggested_follow_up: str = "Next scheduled cycle (7 days)",
        urgency_score: float = 50.0,
        actions: Optional[List[MaintenanceAction]] = None,
        inspection_checklist: Optional[List[str]] = None,
        primary_fault_mode: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        safety_disclaimer: str = MAINTENANCE_DISCLAIMER,
        **kwargs: Any,
    ) -> None:
        self.sensor_id: str = str(sensor_id)
        self.sensor_name: str = str(sensor_name)
        self.priority: str = str(priority)
        self.recommended_action: str = str(recommended_action)
        self.reason: str = str(reason)
        self.supporting_evidence: List[str] = list(supporting_evidence) if supporting_evidence is not None else []
        self.health_metrics: Dict[str, Any] = dict(health_metrics) if health_metrics is not None else {}
        self.confidence_category: str = str(confidence_category)
        self.suggested_follow_up: str = str(suggested_follow_up)
        self.urgency_score: float = float(urgency_score)
        self.actions: List[MaintenanceAction] = list(actions) if actions is not None else []
        self.inspection_checklist: List[str] = list(inspection_checklist) if inspection_checklist is not None else []
        self.primary_fault_mode: Optional[str] = primary_fault_mode
        self.safety_disclaimer: str = str(safety_disclaimer)

        if limitations is not None:
            self.limitations: List[str] = list(limitations)
        else:
            self.limitations = [
                "Recommendations derived from synthetic multi-sensor simulation and topological inference.",
                "Does not replace certified on-site mechanical inspection or safety interlocks.",
                "Telemetry-based anomaly patterns do not predict exact hardware failure dates.",
            ]

    # Aliases for backward compatibility and caller flexibility
    @property
    def sensor(self) -> str:
        return self.sensor_id

    @property
    def health_score(self) -> float:
        return float(self.health_metrics.get("health_score", 0.0))

    @property
    def status_summary(self) -> str:
        return self.reason

    @property
    def reason_for_recommendation(self) -> str:
        return self.reason

    @property
    def evidence(self) -> List[str]:
        return self.supporting_evidence

    @property
    def review_urgency(self) -> str:
        return self.suggested_follow_up

    def __repr__(self) -> str:
        return (
            f"MaintenanceAdvisory(sensor_id={self.sensor_id!r}, priority={self.priority!r}, "
            f"action={self.recommended_action!r}, urgency_score={self.urgency_score:.1f})"
        )


@dataclass
class NetworkMaintenanceSummary:
    """Network-level aggregated maintenance planning summary across all sensors."""
    total_sensors_evaluated: int
    critical_cases: int
    high_priority_cases: int
    medium_priority_cases: int
    low_priority_cases: int
    insufficient_data_cases: int
    sensors_requiring_more_data: List[str]
    highest_priority_sensor: Optional[str]
    overall_maintenance_attention_level: str   # 'Critical', 'High', 'Medium', 'Low', 'Nominal', 'Insufficient Data'
    ranked_action_list: List[Dict[str, Any]] = field(default_factory=list)
    advisories: Dict[str, MaintenanceAdvisory] = field(default_factory=dict)
    disclaimer: str = MAINTENANCE_DISCLAIMER


# Sensor baseline metadata and physics-informed maintenance protocols
SENSOR_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "S1": {
        "name": "Bearing Temperature (S1)",
        "subsystem": "Bearing Housing Assembly",
        "drift": {
            "action_name": "Sensor calibration review",
            "title": "Thermal Runaway & Lubrication Calibration Review",
            "description": "Gradual temperature rise indicates hydrodynamic grease depletion, boundary friction, or RTD zero-point calibration drift.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S1-01",
                    title="Lubricant Quality and Quantity Check",
                    description="Inspect grease condition for oxidation/particulate darkening. Replenish grease with OEM-specified ISO VG lubricant.",
                    component="Bearing Housing Reservoir",
                    required_tools=["Grease gun with meter", "Oil sampling bottle", "Infrared pyrometer"],
                    safety_notes="Verify motor is de-energized or guard plates are intact before accessing grease fittings.",
                ),
                MaintenanceAction(
                    action_id="ACT-S1-02",
                    title="RTD Thermocouple Calibration",
                    description="Perform 2-point temperature calibration on PT100 RTD sensor element and transmitter module.",
                    component="Thermal Well / Transmitter",
                    required_tools=["Dry-block temperature calibrator", "Digital multimeter"],
                    safety_notes="Ensure thermal well process sleeve is depressurized before probe extraction.",
                ),
            ],
            "checklist": [
                "Verify housing surface temperature with calibrated IR thermometer.",
                "Check bearing seal for lubricant egress or contamination ingress.",
                "Inspect motor cooling fan shroud for particulate fouling or debris blockage.",
                "Confirm RTD sensor sheath is fully seated in thermowell.",
            ],
        },
        "spike": {
            "action_name": "Connection and signal-quality inspection",
            "title": "RTD Cable Shielding & Electrical Noise Inspection",
            "description": "High-frequency thermal impulses in high-thermal-mass housings typically indicate instrumentation grounding faults or loose terminal blocks.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S1-03",
                    title="RTD Shielding and Continuity Inspection",
                    description="Inspect cable shield continuity, ground loop integrity, and terminal block screw torque.",
                    component="Signal Cabling & Termination",
                    required_tools=["Insulation resistance tester", "Torque screwdriver"],
                    safety_notes="De-energize 24V loop power before tightening terminal screws.",
                ),
            ],
            "checklist": [
                "Inspect cable run for proximity to high-frequency VFD power cables (EMI).",
                "Verify single-point earth grounding at marshalling cabinet.",
                "Inspect RTD junction head for condensation or corrosion.",
            ],
        },
        "stuck": {
            "action_name": "Sensor replacement assessment",
            "title": "Temperature Transducer Signal Freeze Diagnostic",
            "description": "Constant unchanging temperature signal indicates transmitter DAC lockup, ADC freeze, or failed sensing element.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S1-04",
                    title="Sensor Loop Power Cycle and Transducer Replacement",
                    description="Power cycle the 4-20mA loop transmitter. If output remains frozen, replace sensor head.",
                    component="4-20mA Temperature Transmitter",
                    required_tools=["Loop calibrator", "Replacement PT100 probe"],
                    safety_notes="Follow Lockout/Tagout (LOTO) protocols for instrument rack power.",
                ),
            ],
            "checklist": [
                "Measure loop current with ammeter (verify not clipped at 4.0mA or 20.0mA fault rail).",
                "Perform cold restart on signal conditioner module.",
                "Replace transmitter module if test heat source produces no analog response.",
            ],
        },
    },
    "S2": {
        "name": "Shaft Vibration (S2)",
        "subsystem": "Drive Shaft & Rotating Coupling",
        "drift": {
            "action_name": "Sensor calibration review",
            "title": "Laser Shaft Alignment & Dynamic Balancing Review",
            "description": "Steadily escalating RMS vibration indicates developing rotor unbalance, shaft misalignment, or accelerometer baseline drift.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S2-01",
                    title="Laser Shaft Alignment and Balance Verification",
                    description="Perform precision optical laser alignment on motor-to-pump coupling; conduct single-plane dynamic balance run.",
                    component="Flexible Shaft Coupling",
                    required_tools=["Laser alignment kit", "Vibration analyzer (FFT spectrum)"],
                    safety_notes="Lock out main motor breaker before mounting laser brackets on shaft.",
                ),
                MaintenanceAction(
                    action_id="ACT-S2-02",
                    title="Foundation and Anchor Bolt Torquing",
                    description="Verify baseplate grouting integrity and torque all motor/pump hold-down bolts to spec.",
                    component="Machine Bedplate / Foundation",
                    required_tools=["Calibrated torque wrench (150-300 Nm)"],
                    safety_notes="Inspect for soft-foot condition using dial indicators before final torquing.",
                ),
            ],
            "checklist": [
                "Record 1X, 2X, and harmonic FFT vibration spectrum.",
                "Inspect flexible spider/grid coupling element for elastomer shredding.",
                "Check foundation hold-down bolts for looseness.",
                "Verify accelerometer mounting stud torque (5-8 Nm).",
            ],
        },
        "spike": {
            "action_name": "Connection and signal-quality inspection",
            "title": "Accelerometer Mounting & Cable Strain Inspection",
            "description": "Transient high-amplitude vibration spikes often stem from cable whipping, loose mounting studs, or impact shocks.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S2-03",
                    title="Accelerometer Stud Mounting and Cable Tie-Down",
                    description="Remove sensor, clean mounting surface, apply silicone grease, re-torque mounting stud to 6 Nm, and secure cable with P-clips.",
                    component="Piezoelectric Accelerometer",
                    required_tools=["Torque wrench", "Surface cleaning solvent"],
                    safety_notes="Ensure machine is stopped before working near rotating shaft.",
                ),
            ],
            "checklist": [
                "Check accelerometer cable for pinch points, abrasion, or sharp bends.",
                "Verify BNC/MIL-spec connector is clean, dry, and tight.",
                "Check for intermittent continuity while gently flexing sensor cable.",
            ],
        },
        "stuck": {
            "action_name": "Sensor replacement assessment",
            "title": "Vibration Transducer Replacement Assessment",
            "description": "Zero AC dynamic signal or constant DC output indicates internal crystal fracture, amplifier saturation, or broken signal lead.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S2-04",
                    title="Piezoelectric Transducer Bias Voltage Check and Replacement",
                    description="Verify ICP/IEPE constant-current supply (2-4 mA) and bias voltage (10-12 VDC). Replace accelerometer if defective.",
                    component="IEPE Accelerometer",
                    required_tools=["Multimeter with diode/bias check", "Replacement accelerometer"],
                    safety_notes="Do not short constant-current excitation lines.",
                ),
            ],
            "checklist": [
                "Measure DC bias voltage at signal conditioner.",
                "Tap mounting block lightly to verify dynamic response on oscilloscope/analyzer.",
                "Replace accelerometer if bias voltage is near 0V (short) or 24V (open).",
            ],
        },
    },
    "S3": {
        "name": "Cooling Pressure (S3)",
        "subsystem": "Hydraulic Manifold & Cooling Circuit",
        "drift": {
            "action_name": "Sensor calibration review",
            "title": "Pressure Transmitter Zero-Point & Line Calibration Review",
            "description": "Gradual hydraulic drift points to regulator diaphragm creep, downstream throttling, or strain-gauge zero drift.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S3-01",
                    title="Pressure Transmitter Zero/Span Calibration",
                    description="Isolate transmitter via block-and-bleed manifold. Apply calibrated test pressure using pneumatic hand pump and verify 4-20mA span.",
                    component="Piezoresistive Pressure Transmitter",
                    required_tools=["Pneumatic hand pump", "Digital pressure calibrator", "Fluke 789 ProcessMeter"],
                    safety_notes="Fully bleed hydraulic pressure before disconnecting any instrument fittings.",
                ),
                MaintenanceAction(
                    action_id="ACT-S3-02",
                    title="Cooling Circuit Relief Valve Inspection",
                    description="Inspect and bench-test line pressure relief valve for seat leakage or setpoint drift.",
                    component="Pressure Relief Valve",
                    required_tools=["Test bench rig", "Seat leakage gauge"],
                    safety_notes="Beware of high-pressure fluid spray; wear face shield and safety glasses.",
                ),
            ],
            "checklist": [
                "Inspect pressure impulse lines for fluid leaks or crystalline residue.",
                "Check manifold isolation valves for complete shutoff.",
                "Verify pressure snubber is clean and not clogged with particulates.",
                "Compare gauge reading against independent master reference gauge.",
            ],
        },
        "spike": {
            "action_name": "Connection and signal-quality inspection",
            "title": "Hydraulic Water-Hammer & Snubber Dampener Inspection",
            "description": "High-pressure spikes can cause piping fatigue or signal amplifier clipping. Inspect accumulator and signal integrity.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S3-03",
                    title="Pulsation Dampener & Impulse Snubber Service",
                    description="Check pre-charge pressure in hydraulic accumulator. Replace sintered bronze snubber fitting.",
                    component="Hydraulic Accumulator / Snubber",
                    required_tools=["Nitrogen charging kit", "Pressure gauge"],
                    safety_notes="Use only dry nitrogen to charge accumulator; never use oxygen or compressed shop air.",
                ),
            ],
            "checklist": [
                "Inspect pipe hangers and hydraulic clamp dampers for loose mounting.",
                "Check fast-closing solenoid valves for improper closing ramp rates.",
                "Verify transmitter electrical connector seal against water ingress.",
            ],
        },
        "stuck": {
            "action_name": "Sensor replacement assessment",
            "title": "Pressure Diaphragm Freeze & Transmitter Replacement",
            "description": "Constant static output indicates diaphragm rupture, severe impulse line sedimentation clog, or sensor bridge failure.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S3-04",
                    title="Impulse Line Blowdown and Transmitter Replacement",
                    description="Perform blowdown on impulse tubing to clear sediment. If pressure output remains frozen, replace transmitter.",
                    component="Pressure Transmitter Cell",
                    required_tools=["Blowdown purge kit", "Replacement transmitter"],
                    safety_notes="Follow plant chemical safety procedures when purging coolant lines.",
                ),
            ],
            "checklist": [
                "Bleed manifold vent screws to observe physical fluid pressure release.",
                "Check sensing diaphragm for indentation or particulate crusting.",
                "Verify sensor bridge resistance with ohmmeter.",
            ],
        },
    },
    "S4": {
        "name": "Coolant Flow (S4)",
        "subsystem": "Cooling Circulation & Flow Delivery",
        "drift": {
            "action_name": "Sensor calibration review",
            "title": "Flow Meter Calibration & Turbine Rotor Review",
            "description": "Gradual flow reduction indicates pump impeller wear, strainer particulate buildup, or flow transmitter scale drift.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S4-01",
                    title="Inline Strainer Cleaning and Basket Inspection",
                    description="Isolate flow loop, open duplex strainer basket, remove debris, and inspect mesh for tears.",
                    component="Suction Strainer Basket",
                    required_tools=["Strainer spanner wrench", "Replacement basket gasket"],
                    safety_notes="Verify flow line is depressurized and cooled to < 40°C before opening strainer cover.",
                ),
                MaintenanceAction(
                    action_id="ACT-S4-02",
                    title="Flow Meter In-Situ Calibration Verification",
                    description="Cross-check flow reading with portable clamp-on ultrasonic flow meter at dedicated calibration spool.",
                    component="Flow Meter Spool Piece",
                    required_tools=["Portable transit-time ultrasonic flow meter", "Acoustic couplant"],
                    safety_notes="Clean pipe surface thoroughly before transducer application.",
                ),
            ],
            "checklist": [
                "Record differential pressure across circulation strainer.",
                "Inspect magnetic flow meter liner for scale or pitting.",
                "Check flow control valve position indicator against feedback telemetry.",
                "Confirm straight run piping upstream (10D) and downstream (5D) is unobstructed.",
            ],
        },
        "spike": {
            "action_name": "Connection and signal-quality inspection",
            "title": "Flow Signal Noise & Entrained Air Bubble Inspection",
            "description": "Erratic flow surges often indicate entrained air pockets, cavitation at the pump suction, or pickup coil grounding faults.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S4-03",
                    title="Pump Net Positive Suction Head (NPSH) Verification",
                    description="Measure suction tank level and inlet head. Bleed high-point air vents on cooling distribution manifold.",
                    component="Cooling Pump Suction",
                    required_tools=["Compound suction gauge", "Manual vent keys"],
                    safety_notes="Wear thermal gloves if coolant is hot.",
                ),
            ],
            "checklist": [
                "Inspect suction piping joints for air ingress leaks under vacuum.",
                "Verify flow meter pickup coil shielding is grounded at one end only.",
                "Listen for cavitation crackling sounds at pump volute.",
            ],
        },
        "stuck": {
            "action_name": "Sensor replacement assessment",
            "title": "Turbine Rotor Lockup & Flow Sensor Replacement",
            "description": "Zero or unchanging flow reading during active pump operation indicates mechanical turbine seizure or transmitter electronics failure.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S4-04",
                    title="Turbine Rotor Inspection and Transducer Replacement",
                    description="Remove flow meter sensor assembly from spool. Inspect turbine bearings for particulate seizure. Replace flow sensor if defective.",
                    component="Turbine Flow Transducer",
                    required_tools=["Pipe wrenches", "Replacement flow sensor kit"],
                    safety_notes="Lock out circulation pump motor before line disassembly.",
                ),
            ],
            "checklist": [
                "Spin rotor manually to check for bearing friction or binding.",
                "Inspect pickup sensor magnetic core for metal filings.",
                "Test pulse frequency output at signal conditioner terminal.",
            ],
        },
    },
    "S5": {
        "name": "Motor Current (S5)",
        "subsystem": "Electric Motor Drive & Phase Electronics",
        "drift": {
            "action_name": "Sensor calibration review",
            "title": "Current Transformer (CT) Burden & Motor Load Review",
            "description": "Sustained elevation in motor current indicates mechanical overload, stator winding insulation degradation, or CT ratio calibration shift.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S5-01",
                    title="Motor Insulation Resistance (Megger) Testing",
                    description="Perform 1000V DC insulation resistance test between phase windings and to motor ground frame. Verify > 100 Megaohms.",
                    component="Motor Stator Windings",
                    required_tools=["1000V Megohmmeter (Megger)", "Safety grounding wand"],
                    safety_notes="De-energize and lock out 480V/415V supply breaker; discharge winding capacitance before touching leads.",
                ),
                MaintenanceAction(
                    action_id="ACT-S5-02",
                    title="Current Transformer Burden and Calibration Check",
                    description="Measure secondary circuit burden impedance and verify CT ratio using a secondary current injection test set.",
                    component="Current Transformer (CT) Module",
                    required_tools=["Clamp-on True-RMS ammeter", "Secondary injection tester"],
                    safety_notes="NEVER open-circuit a live CT secondary winding; lethal high voltage will be induced.",
                ),
            ],
            "checklist": [
                "Measure phase current balance across all 3 phases (max allowable imbalance: 3%).",
                "Inspect motor terminal box for darkened or heat-stressed lugs.",
                "Check motor cooling fan cowl and fins for heavy dust buildup.",
                "Record operating power factor and supply voltage.",
            ],
        },
        "spike": {
            "action_name": "Connection and signal-quality inspection",
            "title": "VFD Switching Noise & Phase Contactor Inspection",
            "description": "Current spikes frequently originate from VFD transient switching harmonics, contactor arcing, or Hall-effect sensor lead noise.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S5-03",
                    title="Motor Starter Contactor and VFD Output Inspection",
                    description="Inspect contactor tips for pitting or flashover. Check VFD output dV/dt filter and load reactor.",
                    component="Motor Control Cabinet (MCC)",
                    required_tools=["Arc flash PPE Category 2/4", "True-RMS multimeter"],
                    safety_notes="Wear certified arc flash personal protective equipment (PPE) before opening energized MCC door.",
                ),
            ],
            "checklist": [
                "Inspect Hall-effect current sensor shield grounding.",
                "Check for loose wire connections at power terminal blocks.",
                "Review VFD fault history log for overcurrent or DC bus ripple alarms.",
            ],
        },
        "stuck": {
            "action_name": "Sensor replacement assessment",
            "title": "Current Transducer Core Saturation & Replacement",
            "description": "Unresponsive constant current output indicates Hall-effect IC latch-up, CT secondary burnout, or analog input card failure.",
            "actions": [
                MaintenanceAction(
                    action_id="ACT-S5-04",
                    title="Hall-Effect Transducer De-gaussing and Replacement",
                    description="De-energize circuit, demagnetize core, and test zero-offset voltage. Replace Hall-effect transducer if unrecoverable.",
                    component="Hall-Effect Current Sensor",
                    required_tools=["Replacement current sensor", "Calibrated True-RMS clamp meter"],
                    safety_notes="Verify zero voltage on power bus before touching sensor aperture.",
                ),
            ],
            "checklist": [
                "Verify DC supply voltage to Hall-effect module (+/- 15V or 24V).",
                "Measure secondary analog voltage with independent calibrated DMM.",
                "Replace analog input channel module if sensor signal tests normal at cabinet terminal.",
            ],
        },
    },
}


class MaintenanceAdvisor:
    """Intelligent predictive maintenance recommendation and fleet planning engine."""

    def __init__(
        self,
        critical_health_threshold: float = 65.0,
        warning_health_threshold: float = 80.0,
        anomaly_frequency_critical: float = 0.30,
        anomaly_frequency_high: float = 0.15,
        anomaly_frequency_medium: float = 0.05,
    ) -> None:
        """Initialize MaintenanceAdvisor with configurable decision boundaries.

        Args:
            critical_health_threshold: Health score below which priority is Critical (default: 65.0).
            warning_health_threshold: Health score below which priority is High (default: 80.0).
            anomaly_frequency_critical: Anomaly fraction triggering Critical (default: 0.30).
            anomaly_frequency_high: Anomaly fraction triggering High (default: 0.15).
            anomaly_frequency_medium: Anomaly fraction triggering Medium (default: 0.05).
        """
        self.critical_health_threshold = float(critical_health_threshold)
        self.warning_health_threshold = float(warning_health_threshold)
        self.anomaly_frequency_critical = float(anomaly_frequency_critical)
        self.anomaly_frequency_high = float(anomaly_frequency_high)
        self.anomaly_frequency_medium = float(anomaly_frequency_medium)

    def recommend(
        self,
        sensor: str,
        health_report: Optional[SensorHealthRecord] = None,
        fault_type_hint: Optional[str] = None,
        confidence_level: Optional[str] = "Moderate",
        rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
        scores_df: Optional[pd.DataFrame] = None,
        onsets: Optional[Dict[str, Optional[int]]] = None,
        investigation_report: Optional[Any] = None,
        **kwargs: Any,
    ) -> MaintenanceAdvisory:
        """Formulate an evidence-based maintenance advisory for an individual sensor node.

        Args:
            sensor: Sensor ID (e.g., 'S1', 'S2').
            health_report: Optional SensorHealthRecord from SensorHealthMonitor.
            fault_type_hint: Inferred or explicit fault mode ('drift', 'spike', 'stuck').
            confidence_level: Localization or diagnostic confidence level.
            rankings: Optional root-cause localization rankings list.
            scores_df: Optional anomaly scores DataFrame.
            onsets: Optional detection onsets dictionary.
            investigation_report: Optional InvestigationReport from FaultExplainer.

        Returns:
            MaintenanceAdvisory with prioritized action, evidence, and technician checklist.
        """
        s_id = str(sensor)
        sensor_data = SENSOR_PROTOCOLS.get(s_id, SENSOR_PROTOCOLS["S1"])
        sensor_name = sensor_data["name"]

        # 1. Handle Insufficient Data / Missing Health Information
        if health_report is None or str(getattr(health_report, "health_status", getattr(health_report, "status", ""))).lower() == "insufficient data":
            return MaintenanceAdvisory(
                sensor_id=s_id,
                sensor_name=sensor_name,
                priority="Insufficient Data",
                recommended_action="Additional data collection",
                reason=(
                    f"Sensor {s_id} ({sensor_name}) has insufficient readings or missing telemetry samples. "
                    "Operational condition cannot be reliably evaluated without streaming telemetry."
                ),
                supporting_evidence=[
                    f"Sensor {s_id} telemetry stream has zero or fewer than minimum required valid samples.",
                    "Health evaluation status is 'Insufficient Data'.",
                    "No anomaly scores or sustained alarms could be computed.",
                ],
                health_metrics={
                    "health_score": 0.0,
                    "health_status": "Insufficient Data",
                    "anomaly_frequency": 0.0,
                    "average_anomaly_score": 0.0,
                    "maximum_anomaly_score": 0.0,
                    "detection_onset": None,
                    "recent_trend": "Unknown",
                },
                confidence_category="Indeterminate",
                suggested_follow_up="Immediate data diagnostics",
                urgency_score=25.0,
                actions=[
                    MaintenanceAction(
                        action_id=f"ACT-{s_id}-99",
                        title="Telemetry Channel Diagnostic & Data Collection",
                        description=f"Inspect telemetry connectivity, DAQ communication bus, and cabling for sensor {s_id}.",
                        component="Telemetry & DAQ Interface",
                        required_tools=["Multimeter", "Network cable tester", "DAQ diagnostics software"],
                        safety_notes="Inspect connections with equipment de-energized if near high voltage.",
                    )
                ],
                inspection_checklist=[
                    "Verify physical wiring and power to sensor transmitter.",
                    "Confirm DAQ channel configuration and baud rate.",
                    "Re-run telemetry collection after establishing communication.",
                ],
                primary_fault_mode=None,
                limitations=[
                    "Cannot infer physical degradation without telemetry observations.",
                    "Priority set to 'Insufficient Data' to avoid misleading maintenance dispatch.",
                ],
            )

        # 2. Extract Metrics from Health Report
        health_score = float(getattr(health_report, "health_score", 100.0))
        health_status = str(getattr(health_report, "health_status", getattr(health_report, "status", "Healthy")))
        anom_freq = float(getattr(health_report, "anomaly_frequency", 0.0))
        max_anom = float(getattr(health_report, "maximum_anomaly_score", 0.0))
        avg_anom = float(getattr(health_report, "average_anomaly_score", 0.0))
        onset = getattr(health_report, "detection_onset", None)
        trend = str(getattr(health_report, "recent_trend", "Stable"))

        # Fallback to onsets dict if report onset is None
        if onset is None and onsets and s_id in onsets:
            onset = onsets[s_id]

        # 3. Check Localization Ranking and Suspect Status
        loc_rank: Optional[int] = None
        loc_score: float = 0.0
        loc_breakdown: Dict[str, Any] = {}
        if rankings:
            for rank_idx, (cand, c_score, b_dict) in enumerate(rankings, start=1):
                if cand == s_id:
                    loc_rank = rank_idx
                    loc_score = float(c_score)
                    loc_breakdown = b_dict
                    break

        primary_suspect_arg = kwargs.get("suspect_sensor")
        is_primary_suspect = (loc_rank == 1) or (primary_suspect_arg is not None and primary_suspect_arg == s_id)

        # 4. Infer Fault Signature
        resolved_fault_mode = fault_type_hint
        if not resolved_fault_mode and is_primary_suspect and investigation_report:
            # Check investigation report evidence for mode keywords
            ev_text = " ".join(getattr(investigation_report, "evidence", [])).lower()
            if "spike" in ev_text:
                resolved_fault_mode = "spike"
            elif "stuck" in ev_text or "freeze" in ev_text:
                resolved_fault_mode = "stuck"
            elif "drift" in ev_text or "ramp" in ev_text:
                resolved_fault_mode = "drift"

        if not resolved_fault_mode:
            # Infer from score characteristics:
            # If maximum anomaly is high but average is very low -> spike
            if max_anom >= 0.60 and avg_anom < 0.25:
                resolved_fault_mode = "spike"
            elif max_anom >= 0.50:
                resolved_fault_mode = "drift"
            else:
                resolved_fault_mode = "drift"

        # 5. Dynamic Priority Classification (Critical, High, Medium, Low)
        # Decision Principles:
        # CRITICAL: Critical health status (<65), or strong anomaly evidence (anom_freq >= 0.30 & max_anom >= 0.70),
        #           or primary suspect with high confidence and severe health degradation.
        # HIGH: Warning status, or significant degradation (65-80), or repeated abnormal readings (anom_freq >= 0.15),
        #       or primary suspect with moderate degradation.
        # MEDIUM: Moderate health degradation (80-88), emerging abnormal patterns (anom_freq >= 0.05), or degrading trend.
        # LOW: Healthy condition (>=88), no significant anomaly evidence (anom_freq < 0.05).
        
        is_high_conf = str(confidence_level).upper() in ("HIGH", "STRONG")

        if (
            health_status == "Critical"
            or health_score < self.critical_health_threshold
            or (anom_freq >= self.anomaly_frequency_critical and max_anom >= 0.70)
            or (is_primary_suspect and is_high_conf and health_score < 75.0)
        ):
            priority = "Critical"
            urgency_score = max(80.0, 100.0 - health_score)
            suggested_follow_up = "Immediate (< 4 hours)"
        elif (
            health_status == "Warning"
            or health_score < self.warning_health_threshold
            or anom_freq >= self.anomaly_frequency_high
            or (max_anom >= 0.65 and health_score < 85.0)
            or (is_primary_suspect and health_score < 85.0)
        ):
            priority = "High"
            urgency_score = max(55.0, 100.0 - health_score)
            suggested_follow_up = "Within 24 hours"
        elif (
            health_score < 88.0
            or anom_freq >= self.anomaly_frequency_medium
            or trend == "Degrading"
            or (loc_rank is not None and loc_rank <= 2 and health_score < 92.0)
        ):
            priority = "Medium"
            urgency_score = max(30.0, 100.0 - health_score)
            suggested_follow_up = "Next scheduled cycle (7 days)"
        else:
            priority = "Low"
            urgency_score = max(5.0, 100.0 - health_score)
            suggested_follow_up = "Routine (30 days)"

        # 6. Recommendation Category Matching Available Evidence
        mode_protocol = sensor_data.get(resolved_fault_mode, sensor_data["drift"])

        if priority == "Critical":
            if resolved_fault_mode == "drift":
                recommended_action = "Sensor calibration review"
            elif resolved_fault_mode == "spike":
                recommended_action = "Connection and signal-quality inspection"
            elif resolved_fault_mode == "stuck":
                recommended_action = "Sensor replacement assessment"
            else:
                recommended_action = "Sensor inspection"
            reason = (
                f"Critical operational degradation detected on {sensor_name} (Health: {health_score:.1f}/100, "
                f"Peak Anomaly: {max_anom:.3f}). Immediate maintenance review required to prevent cascade."
            )
        elif priority == "High":
            if resolved_fault_mode == "drift":
                recommended_action = "Sensor calibration review"
            elif resolved_fault_mode == "spike":
                recommended_action = "Connection and signal-quality inspection"
            elif resolved_fault_mode == "stuck":
                recommended_action = "Sensor replacement assessment"
            else:
                recommended_action = "Sensor inspection"
            reason = (
                f"Significant telemetry anomaly pattern on {sensor_name} (Health: {health_score:.1f}/100, "
                f"Anomaly Freq: {anom_freq*100:.1f}%). Scheduled inspection recommended within 24 hours."
            )
        elif priority == "Medium":
            recommended_action = "Increased monitoring"
            reason = (
                f"Moderate degradation or emerging anomalous deviation on {sensor_name} "
                f"(Health: {health_score:.1f}/100, Trend: {trend}). Increased monitoring and inspection recommended."
            )
        else:
            recommended_action = "Routine maintenance review"
            reason = (
                f"Sensor {sensor_name} is operating within nominal parameters (Health: {health_score:.1f}/100). "
                "Maintain standard preventive maintenance schedule."
            )

        # 7. Formulate Supporting Evidence List
        evidence: List[str] = [
            f"Sensor operational health score evaluated at {health_score:.1f} / 100 ({health_status}).",
            f"Observed anomaly frequency of {anom_freq * 100:.1f}% with peak anomaly score of {max_anom:.3f} and mean {avg_anom:.3f}.",
        ]

        if onset is not None:
            evidence.append(f"Confirmed alarm onset sustained from timestep t = {onset}.")
        else:
            evidence.append("No sustained alarm onset threshold was crossed during the monitoring period.")

        if loc_rank is not None:
            evidence.append(f"Root-cause localization ranking: #{loc_rank} with composite score of {loc_score:.4f}.")
            if is_primary_suspect:
                evidence.append("Primary root-cause suspect based on onset earliness and physical mesh coupling propagation.")
        else:
            evidence.append("Sensor was not flagged as a primary root-cause suspect.")

        if trend != "Stable":
            evidence.append(f"Recent operational trajectory displays a {trend.lower()} trend.")

        # Determine confidence category
        if priority in ("Critical", "High") and is_primary_suspect:
            confidence_cat = str(confidence_level).capitalize() if confidence_level else "High"
        elif priority == "Low":
            confidence_cat = "High"
        elif priority == "Medium":
            confidence_cat = "Moderate"
        else:
            confidence_cat = "Moderate"

        # If this is the primary suspect sensor, prioritize action urgency
        if is_primary_suspect and priority != "Low":
            urgency_score = min(100.0, urgency_score + 10.0)

        # Action items and checklists from domain protocols
        actions = mode_protocol.get("actions", [])
        checklist = mode_protocol.get("checklist", [])

        # If priority is Low, adapt checklist for routine checks
        if priority == "Low":
            checklist = [
                f"Verify {sensor_name} electrical connection and grounding.",
                f"Confirm zero baseline stability ({health_report.baseline_value} {health_report.unit}).",
                "Ensure protective conduit and mounting bracket remain secure.",
            ]
            actions = [
                MaintenanceAction(
                    action_id=f"ACT-{s_id}-ROUTINE",
                    title="Routine Preventive Telemetry Check",
                    description=f"Standard periodic inspection of {sensor_name} wiring, baseline calibration, and physical environment.",
                    component=sensor_data["subsystem"],
                    required_tools=["Multimeter", "Visual inspection checklist"],
                    safety_notes="Follow standard plant electrical safety guidelines.",
                )
            ]

        health_metrics_dict = {
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "anomaly_frequency": round(anom_freq, 4),
            "average_anomaly_score": round(avg_anom, 4),
            "maximum_anomaly_score": round(max_anom, 4),
            "detection_onset": onset,
            "recent_trend": trend,
            "localization_rank": loc_rank,
            "localization_score": round(loc_score, 4),
        }

        return MaintenanceAdvisory(
            sensor_id=s_id,
            sensor_name=sensor_name,
            priority=priority,
            recommended_action=recommended_action,
            reason=reason,
            supporting_evidence=evidence,
            health_metrics=health_metrics_dict,
            confidence_category=confidence_cat,
            suggested_follow_up=suggested_follow_up,
            urgency_score=round(urgency_score, 1),
            actions=actions,
            inspection_checklist=checklist,
            primary_fault_mode=resolved_fault_mode if priority != "Low" else None,
            safety_disclaimer=MAINTENANCE_DISCLAIMER,
        )

    def advise_fleet(
        self,
        suspect_sensor: Optional[str] = None,
        health_reports: Optional[Dict[str, Any]] = None,
        fault_type: Optional[str] = None,
        confidence_level: Optional[str] = None,
        rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
        investigation_report: Optional[Any] = None,
        scores_df: Optional[pd.DataFrame] = None,
        onsets: Optional[Dict[str, Optional[int]]] = None,
        **kwargs: Any,
    ) -> Dict[str, MaintenanceAdvisory]:
        """Generate fleet-wide maintenance advisories across all monitored sensors.

        Handles flexible argument orders (supporting legacy calls where health_reports is first arg).

        Returns:
            Dict mapping sensor ID (e.g. 'S1'..'S5') to MaintenanceAdvisory.
        """
        # Support flexible argument orders
        if suspect_sensor is not None and not isinstance(suspect_sensor, str) and isinstance(suspect_sensor, dict):
            # Positional swap: advise_fleet(health_reports, ...)
            reports_dict = suspect_sensor
            primary_suspect = kwargs.get("suspect_sensor")
        else:
            primary_suspect = suspect_sensor
            reports_dict = health_reports or kwargs.get("health_reports", {})

        if not primary_suspect and rankings:
            primary_suspect = rankings[0][0]
        elif not primary_suspect and investigation_report:
            primary_suspect = getattr(investigation_report, "suspected_sensor", None)

        advisories: Dict[str, MaintenanceAdvisory] = {}
        all_sensor_keys = ["S1", "S2", "S3", "S4", "S5"]

        # Ensure all 5 standard sensors are addressed even if missing in reports_dict
        sensors_to_evaluate = sorted(list(set(all_sensor_keys + list(reports_dict.keys()))))

        for s_id in sensors_to_evaluate:
            report = reports_dict.get(s_id)
            f_type = fault_type if s_id == primary_suspect else None
            conf = confidence_level if confidence_level else "Moderate"

            advisory = self.recommend(
                sensor=s_id,
                health_report=report,
                fault_type_hint=f_type,
                confidence_level=conf,
                rankings=rankings,
                scores_df=scores_df,
                onsets=onsets,
                investigation_report=investigation_report,
                suspect_sensor=primary_suspect,
            )
            advisories[s_id] = advisory

        return advisories

    def summarize_network_maintenance(
        self,
        advisories: Dict[str, MaintenanceAdvisory],
    ) -> NetworkMaintenanceSummary:
        """Aggregate individual sensor advisories into a cohesive network maintenance plan.

        Args:
            advisories: Dict of sensor ID to MaintenanceAdvisory.

        Returns:
            NetworkMaintenanceSummary with counts, attention level, and ranked actions.
        """
        total = len(advisories)
        crit_count = 0
        high_count = 0
        med_count = 0
        low_count = 0
        insufficient_count = 0
        insufficient_sensors: List[str] = []

        # Ranking priority weight mapping
        priority_weights = {
            "Critical": 400.0,
            "High": 300.0,
            "Medium": 200.0,
            "Low": 100.0,
            "Insufficient Data": 50.0,
        }

        ranked_items = []

        for s_id, adv in advisories.items():
            p = adv.priority
            if p == "Critical":
                crit_count += 1
            elif p == "High":
                high_count += 1
            elif p == "Medium":
                med_count += 1
            elif p == "Low":
                low_count += 1
            elif p == "Insufficient Data":
                insufficient_count += 1
                insufficient_sensors.append(s_id)

            suspect_bonus = 15.0 if (getattr(adv, "primary_fault_mode", None) and adv.priority != "Low") else 0.0
            composite_rank_score = priority_weights.get(p, 0.0) + adv.urgency_score + suspect_bonus
            ranked_items.append((composite_rank_score, adv))

        # Sort descending by priority and urgency
        ranked_items.sort(key=lambda item: item[0], reverse=True)

        ranked_action_list = []
        highest_priority_sensor = ranked_items[0][1].sensor_id if ranked_items else None

        for rank_idx, (_, adv) in enumerate(ranked_items, start=1):
            ranked_action_list.append({
                "rank": rank_idx,
                "sensor_id": adv.sensor_id,
                "sensor_name": adv.sensor_name,
                "priority": adv.priority,
                "recommended_action": adv.recommended_action,
                "urgency_score": adv.urgency_score,
                "follow_up": adv.suggested_follow_up,
                "reason": adv.reason,
            })

        # Determine overall maintenance attention level
        if crit_count > 0:
            overall_attention = "Critical"
        elif high_count > 0:
            overall_attention = "High"
        elif med_count > 0:
            overall_attention = "Medium"
        elif insufficient_count == total and total > 0:
            overall_attention = "Insufficient Data"
        else:
            overall_attention = "Nominal"

        return NetworkMaintenanceSummary(
            total_sensors_evaluated=total,
            critical_cases=crit_count,
            high_priority_cases=high_count,
            medium_priority_cases=med_count,
            low_priority_cases=low_count,
            insufficient_data_cases=insufficient_count,
            sensors_requiring_more_data=sorted(insufficient_sensors),
            highest_priority_sensor=highest_priority_sensor,
            overall_maintenance_attention_level=overall_attention,
            ranked_action_list=ranked_action_list,
            advisories=advisories,
            disclaimer=MAINTENANCE_DISCLAIMER,
        )

    # Alias for method name consistency
    plan_network_maintenance = summarize_network_maintenance
