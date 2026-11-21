"""Layout assembly module for the FaultMesh AI Streamlit dashboard.

Responsibility:
    Assembles modular dashboard layout across eight core functional sections:
    Overview, Sensor Monitoring, Network Topology, Fault Injection, Anomaly Detection,
    Root-Cause Localization, Benchmark Evaluation, and Export Reports.
    Coordinates controls, displays, and components without executing simulation logic.
"""

from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
import pandas as pd

from dashboard.components import (
    render_kpi_cards,
    render_sensor_timeseries,
    render_network_graph,
    render_localization_table,
    render_benchmark_summary,
    render_download_controls,
)
from dashboard.utils import SimConfig, SENSORS

# Sensor baseline specifications from PROJECT_RULES.md
SENSOR_SPECIFICATIONS = [
    {"Sensor ID": "S1", "Parameter": "Temperature", "Baseline": "60.0 °C", "Noise σ (1.5%)": "0.90 °C", "Role": "Operating Heat"},
    {"Sensor ID": "S2", "Parameter": "Vibration", "Baseline": "2.0 mm/s", "Noise σ (1.5%)": "0.030 mm/s", "Role": "Mechanical Transducer"},
    {"Sensor ID": "S3", "Parameter": "Pressure", "Baseline": "5.0 bar", "Noise σ (1.5%)": "0.075 bar", "Role": "Hydraulic/Fluid Loop"},
    {"Sensor ID": "S4", "Parameter": "Flow", "Baseline": "30.0 L/min", "Noise σ (1.5%)": "0.450 L/min", "Role": "Circulation Flow Meter"},
    {"Sensor ID": "S5", "Parameter": "Current", "Baseline": "10.0 A", "Noise σ (1.5%)": "0.150 A", "Role": "Motor Drive Current"},
]


def render_header(st_ctx: Any = st) -> None:
    """Render top banner, title, and mandatory prototype disclaimer."""
    st_ctx.title("FaultMesh AI -- Intelligent Cooperative IoT Fault Localization")
    st_ctx.markdown(
        "**Graph-guided root cause isolation** across coupled multi-sensor industrial mesh networks. "
        "Separates originating fault sources from delayed downstream cascades using topological inference."
    )
    st_ctx.info(
        "**Simulated IoT Monitoring Prototype** -- Data generated via synthetic physical simulation. "
        "Not connected to physical hardware sensors."
    )


def sidebar_controls(st_ctx: Any = st) -> SimConfig:
    """Render sidebar simulation and fault injection configuration controls.

    Ensures fault_start is strictly less than total_steps, and spike duration
    does not exceed the remaining simulation window.

    Args:
        st_ctx: Streamlit context or module.

    Returns:
        SimConfig dataclass instance containing all user-configured parameters.
    """
    st_ctx.sidebar.header("Simulation Controls")

    # Random seed
    seed = st_ctx.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
        help="Deterministic seed for Gaussian baseline noise.",
    )

    # Simulation duration
    st_ctx.sidebar.subheader("Simulation Window")
    total_steps = st_ctx.sidebar.slider(
        "Total Simulation Steps",
        min_value=200,
        max_value=500,
        value=300,
        step=10,
        help="Total time steps generated in the simulation stream.",
    )

    # Fault configuration
    st_ctx.sidebar.subheader("Fault Injection Parameters")
    fault_sensor = st_ctx.sidebar.selectbox(
        "Target Sensor (Ground Truth)",
        options=SENSORS,
        index=2,  # Default S3
        help="Root-cause sensor to inject fault into.",
    )

    fault_type = st_ctx.sidebar.selectbox(
        "Fault Type",
        options=["drift", "spike", "stuck"],
        index=0,  # Default drift
        help="drift = gradual ramp, spike = transient impulse, stuck = frozen constant value.",
    )

    fault_magnitude_ratio = st_ctx.sidebar.slider(
        "Fault Magnitude Ratio (relative to baseline)",
        min_value=0.1,
        max_value=1.5,
        value=0.5,
        step=0.05,
        help="Physical offset scaled relative to target sensor baseline.",
    )

    # Validation: Ensure fault_start < total_steps
    max_start = max(100, int(total_steps) - 1)
    fault_start = st_ctx.sidebar.slider(
        "Fault Start Timestep",
        min_value=100,
        max_value=max_start,
        value=min(150, max_start),
        step=5,
        help="Timestep when fault begins (first 100 steps reserved for normal training).",
    )

    # Validation: Ensure spike duration does not extend beyond available simulation window
    remaining_steps = max(1, int(total_steps) - int(fault_start))
    spike_duration = 10
    if fault_type == "spike":
        max_spike = min(30, remaining_steps)
        spike_duration = st_ctx.sidebar.slider(
            "Spike Duration (steps)",
            min_value=1,
            max_value=max_spike,
            value=min(10, max_spike),
            step=1,
            help="Number of steps the transient impulse persists.",
        )
    else:
        spike_duration = min(10, remaining_steps)

    st_ctx.sidebar.markdown("---")
    st_ctx.sidebar.caption("Prototype Version 1.0.0 | FaultMesh AI")

    return SimConfig(
        seed=int(seed),
        fault_sensor=str(fault_sensor),
        fault_type=str(fault_type),
        fault_magnitude_ratio=float(fault_magnitude_ratio),
        fault_start=int(fault_start),
        spike_duration=int(spike_duration),
        total_steps=int(total_steps),
        noise_ratio=0.015,
        coupling_strength=0.40,
        coupling_delay=2,
    )


def build_overview(
    st_ctx: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    onsets: Optional[Dict[str, Optional[int]]],
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Build Section 1: Overview with KPI cards and architecture summary."""
    st_ctx.subheader("System Overview & Live KPI Metrics")
    render_kpi_cards(st_ctx, readings, scores, onsets, rankings)

    st_ctx.markdown("### Operational Summary")
    if rankings and len(rankings) > 0:
        top_sensor, top_score, top_bd = rankings[0]
        st_ctx.info(
            f"**Diagnostic Status:** Active anomalous event detected. "
            f"Sensor **{top_sensor}** is flagged as the primary root cause (Composite Score: **{top_score:.4f}**). "
            f"Neighboring nodes reflect delayed secondary propagation consistent with the 40% coupling model."
        )
    else:
        st_ctx.info(
            "**Diagnostic Status:** System nominal or simulation pending. "
            "Click **Run Simulation** in the sidebar to execute multi-sensor simulation and localization."
        )

    # System Architecture Highlights
    col1, col2 = st_ctx.columns(2)
    with col1:
        st_ctx.markdown("#### Detection Architecture")
        st_ctx.markdown(
            "- **Algorithm:** Isolation Forest trained on clean baseline samples (t = 0..99).\n"
            "- **Engineered Features:** Rolling Mean, Rolling Volatility, Delta Differences, Standardized Z-Scores.\n"
            "- **Alarm Filter:** Score threshold >= 0.60 sustained for >= 3 consecutive timesteps."
        )
    with col2:
        st_ctx.markdown("#### Localization Multi-Criteria Weights")
        st_ctx.markdown(
            "- **Own Anomaly Score:** 35% (Magnitude of intrinsic sensor abnormality)\n"
            "- **Detection Earliness:** 30% (Relative onset timing preceding coupled cascades)\n"
            "- **Neighbor Support:** 20% (Delayed downstream anomalies on connected peers)\n"
            "- **Coverage:** 15% (Local topological influence density across mesh)"
        )


def build_sensor_monitoring(
    st_ctx: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
) -> None:
    """Build Section 2: Sensor Monitoring with interactive time-series plots."""
    st_ctx.subheader("Multi-Sensor Real-Time Stream Monitoring")

    if readings is None or readings.empty:
        st_ctx.warning("No simulation readings available. Run simulation to populate streams.")
        return

    available_sensors = [c for c in readings.columns if c not in ("time", "timestep") and not c.startswith("_")]
    selected_sensors = st_ctx.multiselect(
        "Select Sensor Streams to Visualize",
        options=available_sensors,
        default=available_sensors,
        help="Filter visible channels in the charts below.",
    )

    render_sensor_timeseries(st_ctx, readings, scores, selected_sensors=selected_sensors)

    # Reference Sensor Baseline Specs
    with st_ctx.expander("Sensor Network Baseline Reference Specifications"):
        st_ctx.dataframe(pd.DataFrame(SENSOR_SPECIFICATIONS), use_container_width=True, hide_index=True)


def build_network_topology(
    st_ctx: Any,
    network_graph: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
    onsets: Optional[Dict[str, Optional[int]]],
) -> None:
    """Build Section 3: Network Topology with interactive graph visualization."""
    st_ctx.subheader("Sensor Mesh Network Physical Topology")
    st_ctx.markdown(
        "Interactive topological map of physical coupling paths. "
        "The highest-ranked root-cause suspect is highlighted in **Red/Diamond**, "
        "alarmed coupled peers in **Amber**, and nominal nodes in **Cyan**."
    )

    render_network_graph(st_ctx, network_graph, rankings, onsets)

    col1, col2 = st_ctx.columns(2)
    with col1:
        st_ctx.markdown("#### Physical Coupling Rules")
        st_ctx.markdown(
            "- **Coupling Strength:** 40% of anomalous deviation transmits to directly connected peers.\n"
            "- **Propagation Delay:** 2 time steps delay before neighbor reflects coupled shift.\n"
            "- **Coupling Formula:** `coupled_dev = 0.40 * (src_dev / src_baseline) * peer_baseline`"
        )
    with col2:
        st_ctx.markdown("#### Undirected Mesh Connections")
        st_ctx.markdown(
            "- **S3 (Pressure):** Connected to S1 (Temp), S2 (Vibration), S4 (Flow), S5 (Current)\n"
            "- **S1 (Temperature):** Connected to S2 (Vibration), S3 (Pressure)\n"
            "- **S2 (Vibration):** Connected to S1 (Temp), S3 (Pressure), S5 (Current)\n"
            "- **S4 (Flow):** Connected to S3 (Pressure), S5 (Current)\n"
            "- **S5 (Current):** Connected to S2 (Vibration), S3 (Pressure), S4 (Flow)"
        )


def build_fault_injection_display(
    st_ctx: Any,
    cfg: Optional[SimConfig],
    fault_dict: Optional[Dict[str, Any]],
    readings: Optional[pd.DataFrame] = None,
) -> None:
    """Build Section 4: Fault Injection parameters and physical impact."""
    st_ctx.subheader("Fault Injection & Ground-Truth Specification")

    st_ctx.markdown(
        "> **Strict Ground-Truth Isolation:** Injected fault labels are maintained strictly for "
        "post-diagnostic validation and are **never** passed into detection or localization algorithms."
    )

    if fault_dict:
        sensor = fault_dict.get("sensor", "N/A")
        ftype = fault_dict.get("type", "N/A")
        start = fault_dict.get("start", "N/A")
        mag = fault_dict.get("magnitude", 0.0)
        dur = fault_dict.get("duration", "Continuous")

        col1, col2, col3, col4, col5 = st_ctx.columns(5)
        col1.metric("Injected Sensor", f"{sensor}")
        col2.metric("Fault Mode", f"{ftype}".capitalize())
        col3.metric("Onset Timestep", f"t = {start}")
        col4.metric("Physical Offset", f"+{mag:.2f}")
        col5.metric("Duration", f"{dur} steps" if dur != "Continuous" else "Continuous")

        st_ctx.markdown("#### Fault Mode Characterization")
        if ftype == "drift":
            st_ctx.write(
                f"**Sensor {sensor}** undergoes a continuous linear drift beginning at timestep `{start}`, "
                f"ramping from nominal baseline to a maximum deviation of `+{mag:.2f}` units."
            )
        elif ftype == "spike":
            st_ctx.write(
                f"**Sensor {sensor}** experiences a sudden transient impulse of amplitude `+{mag:.2f}` units "
                f"starting at timestep `{start}` and persisting for `{dur}` steps before returning to normal baseline."
            )
        elif ftype == "stuck":
            st_ctx.write(
                f"**Sensor {sensor}** freezes at timestep `{start}` at a fixed value of baseline `+{mag:.2f}`, "
                "with noise variance dropping completely to zero."
            )
    else:
        st_ctx.info("No active fault injected in this simulation run (Clean Nominal Baseline).")


def build_anomaly_detection(
    st_ctx: Any,
    scores: Optional[pd.DataFrame],
    onsets: Optional[Dict[str, Optional[int]]],
) -> None:
    """Build Section 5: Anomaly Detection evaluation and onset timing."""
    st_ctx.subheader("Anomaly Detection & Alarm Onset Timing")

    st_ctx.markdown(
        "Anomaly detection identifies abnormal deviations on each sensor stream using an ensemble of "
        "Isolation Forest models and standardized Z-scores. Alarms require 3 consecutive readings above threshold 0.60."
    )

    if not onsets:
        st_ctx.info("No anomaly onset data available. Run simulation to execute detector.")
        return

    # Onset Status Table
    onset_rows = []
    valid_onsets = [t for t in onsets.values() if t is not None]
    earliest_t = min(valid_onsets) if valid_onsets else None

    for sensor in SENSORS:
        t = onsets.get(sensor)
        if t is not None:
            if t == earliest_t:
                status = "EARLIEST ALARM"
            else:
                delay = t - earliest_t if earliest_t is not None else 0
                status = f"Coupled Alarm (+{delay} steps delayed)"
            onset_str = f"t = {t}"
        else:
            status = "Nominal (No Alarm Triggered)"
            onset_str = "None"

        onset_rows.append({
            "Sensor ID": sensor,
            "Confirmed Detection Onset": onset_str,
            "Detection Status": status,
        })

    st_ctx.dataframe(pd.DataFrame(onset_rows), use_container_width=True, hide_index=True)


def build_localization(
    st_ctx: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
    onsets: Optional[Dict[str, Optional[int]]] = None,
) -> None:
    """Build Section 6: Root-Cause Localization rankings and explanations."""
    st_ctx.subheader("Graph-Based Root-Cause Localization Rankings")

    st_ctx.markdown(
        "Candidate sensors are ranked in descending order of composite root-cause score. "
        "The algorithm rewards early alarms and strong delayed propagation across neighboring peers."
    )

    render_localization_table(st_ctx, rankings)


def build_benchmark_evaluation(
    st_ctx: Any,
    benchmark: Optional[Dict[str, Any]],
) -> None:
    """Build Section 7: Benchmark Evaluation performance reporting."""
    st_ctx.subheader("Systematic Evaluation Benchmark Results")
    render_benchmark_summary(st_ctx, benchmark)


def build_export_section(
    st_ctx: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Build Section 8: Export Reports and CSV downloads."""
    st_ctx.subheader("Export Reports & Data Artifacts")
    st_ctx.markdown(
        "Download simulation readings, calibrated multivariate anomaly scores, "
        "and localization diagnostic reports for auditing and verification."
    )

    render_download_controls(st_ctx, readings, scores, rankings)


def render_dashboard_layout(
    st_ctx: Any,
    cfg: SimConfig,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    onsets: Optional[Dict[str, Optional[int]]],
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
    network_graph: Any,
    benchmark: Optional[Dict[str, Any]],
) -> None:
    """Assemble complete 8-section tabbed dashboard layout.

    Args:
        st_ctx: Streamlit context.
        cfg: Active simulation configuration.
        readings: Raw simulated sensor readings DataFrame.
        scores: Calibrated anomaly scores DataFrame.
        onsets: Detection onsets dictionary per sensor.
        rankings: Ranked root-cause suspect tuples.
        network_graph: Sensor network graph topology.
        benchmark: Benchmark evaluation metrics dictionary.
    """
    render_header(st_ctx)

    tabs = st_ctx.tabs([
        "Overview",
        "Sensor Monitoring",
        "Network Topology",
        "Fault Injection",
        "Anomaly Detection",
        "Root-Cause Localization",
        "Benchmark Evaluation",
        "Export Reports",
    ])

    with tabs[0]:
        build_overview(st_ctx, readings, scores, onsets, rankings)

    with tabs[1]:
        build_sensor_monitoring(st_ctx, readings, scores)

    with tabs[2]:
        build_network_topology(st_ctx, network_graph, rankings, onsets)

    with tabs[3]:
        fault_dict = readings.attrs.get("fault") if (readings is not None and hasattr(readings, "attrs")) else None
        build_fault_injection_display(st_ctx, cfg, fault_dict, readings=readings)

    with tabs[4]:
        build_anomaly_detection(st_ctx, scores, onsets)

    with tabs[5]:
        build_localization(st_ctx, rankings, onsets=onsets)

    with tabs[6]:
        build_benchmark_evaluation(st_ctx, benchmark)

    with tabs[7]:
        build_export_section(st_ctx, readings, scores, rankings)
