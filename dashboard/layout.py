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
    render_explainable_investigation,
    render_sensor_health_center,
    render_predictive_maintenance_center,
    render_executive_reports_tab,
    render_live_comm_stats,
    render_live_sensor_cards,
    render_live_diagnostic_transparency_table,
    render_calibration_panel,
)
from dashboard.utils import SimConfig, SENSORS
from faultmesh.ingestion import SENSOR_DEFINITIONS, STANDARD_UNITS

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
    latest_readings: Optional[Dict[str, Dict[str, Any]]] = None,
    sensor_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
    readings_df: Optional[pd.DataFrame] = None,
) -> None:
    """Build Section 5: Anomaly Detection evaluation and multi-sensor alarm monitoring."""
    st_ctx.subheader("Anomaly Detection & Alarm Onset Timing")

    st_ctx.markdown(
        "Anomaly detection identifies abnormal deviations on each sensor stream using an ensemble of "
        "Isolation Forest models and standardized Z-scores. Alarms require 3 consecutive readings above threshold 0.60."
    )

    valid_onsets = [t for t in (onsets.values() if onsets else []) if t is not None]
    earliest_t = min(valid_onsets) if valid_onsets else None

    # Determine data sufficiency
    n_samples = len(readings_df) if readings_df is not None else (len(scores) if scores is not None else 0)

    rows = []
    for sensor in SENSORS:
        diag = sensor_diagnostics.get(sensor, {}) if sensor_diagnostics else {}
        r_info = latest_readings.get(sensor, {}) if latest_readings else {}

        # 1. Sensor ID & Name
        s_name = diag.get("name") or SENSOR_DEFINITIONS.get(sensor, {}).get("name", sensor)
        s_id_display = f"{sensor} ({s_name})"

        # 2. Latest Reading and Unit
        val = diag.get("latest_value")
        if val is None:
            val = r_info.get("value")
        if val is None and readings_df is not None and not readings_df.empty and sensor in readings_df.columns:
            val = float(readings_df[sensor].iloc[-1])
        unit = diag.get("unit") or r_info.get("unit") or STANDARD_UNITS.get(sensor, "")
        val_display = f"{val:.2f} {unit}" if isinstance(val, (int, float)) else "Awaiting Data"

        # 3. Anomaly Score
        score = diag.get("anomaly_score")
        if score is None and scores is not None and not scores.empty and sensor in scores.columns:
            score = float(scores[sensor].iloc[-1])
        score_val = float(score) if isinstance(score, (int, float)) else 0.0
        score_display = f"{score_val:.4f}"

        # 4. Detection Status
        t = onsets.get(sensor) if onsets else None
        if t is not None:
            if t == earliest_t:
                status = f"🚨 EARLIEST ALARM (Onset t={t})"
            else:
                delay = t - earliest_t if earliest_t is not None else 0
                status = f"⚠️ Coupled Alarm (+{delay} delayed, t={t})"
        elif score_val >= 0.60:
            status = f"⚡ Elevated Transient (Score {score_val:.3f} > 0.60)"
        elif n_samples < 10 and not val:
            status = "⏳ Insufficient Data"
        else:
            status = "✅ Nominal (No Alarm Triggered)"

        # 5. Detection Timestamp
        ts_onset = f"t = {t}" if t is not None else "None (Nominal)"
        ts_live = diag.get("timestamp") or r_info.get("timestamp")
        if ts_live and ts_live != "N/A":
            ts_display = f"{ts_onset} ({ts_live})"
        else:
            ts_display = ts_onset

        # 6. Data Sufficiency Status
        sufficiency = diag.get("data_sufficiency")
        if not sufficiency:
            sufficiency = f"Sufficient ({n_samples} samples)" if n_samples >= 15 else f"Collecting ({n_samples}/15 samples)"

        rows.append({
            "Sensor ID": s_id_display,
            "Latest Sensor Reading & Unit": val_display,
            "Anomaly Score": score_display,
            "Detection Status": status,
            "Detection Timestamp": ts_display,
            "Data Sufficiency Status": sufficiency,
        })

    st_ctx.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Anomaly Scores Timeseries Visualization
    if scores is not None and not scores.empty and any(s in scores.columns for s in SENSORS):
        st_ctx.markdown("#### Calibrated Anomaly Scores Over Time (Cutoff: 0.60)")
        import plotly.graph_objects as go
        fig = go.Figure()
        time_axis = scores["time"] if "time" in scores.columns else (scores["timestep"] if "timestep" in scores.columns else list(range(len(scores))))
        colors = {"S1": "#2563EB", "S2": "#7C3AED", "S3": "#DC2626", "S4": "#D97706", "S5": "#059669"}
        for s in SENSORS:
            if s in scores.columns:
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=scores[s],
                    mode="lines",
                    name=f"{s} ({SENSOR_DEFINITIONS.get(s, {}).get('name', s)})",
                    line=dict(color=colors.get(s, "#888888"), width=2),
                ))
        fig.add_hline(
            y=0.60,
            line_dash="dash",
            line_color="rgba(220, 38, 38, 0.85)",
            annotation_text="Alarm Cutoff (0.60)",
            annotation_position="top right",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            yaxis=dict(range=[0.0, 1.05], title="Anomaly Score [0, 1]"),
            xaxis=dict(title="Sample Step"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st_ctx.plotly_chart(fig, use_container_width=True)


def build_localization(
    st_ctx: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
    onsets: Optional[Dict[str, Optional[int]]] = None,
    diagnosis_status: Optional[str] = None,
    primary_suspect: Optional[str] = None,
) -> None:
    """Build Section 6: Root-Cause Localization rankings and explanations."""
    st_ctx.subheader("Graph-Based Root-Cause Localization Rankings")

    st_ctx.markdown(
        "Candidate sensors are ranked in descending order of composite root-cause score. "
        "The algorithm rewards early alarms and strong delayed propagation across neighboring peers."
    )

    render_localization_table(st_ctx, rankings, diagnosis_status=diagnosis_status, primary_suspect=primary_suspect)


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
    investigation_report: Optional[Any] = None,
    health_reports: Optional[Dict[str, Any]] = None,
    network_health: Optional[Any] = None,
    advisories: Optional[Dict[str, Any]] = None,
    maintenance_summary: Optional[Any] = None,
    investigation_summary: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Assemble complete tabbed dashboard layout including Phase 1, 2, 3 & 4 sections.

    Args:
        st_ctx: Streamlit context.
        cfg: Active simulation configuration.
        readings: Raw simulated sensor readings DataFrame.
        scores: Calibrated anomaly scores DataFrame.
        onsets: Detection onsets dictionary per sensor.
        rankings: Ranked root-cause suspect tuples.
        network_graph: Sensor network graph topology.
        benchmark: Benchmark evaluation metrics dictionary.
        investigation_report: Optional InvestigationReport from FaultExplainer (Phase 1).
        health_reports: Optional Dict of SensorHealthRecord from SensorHealthMonitor (Phase 2).
        network_health: Optional NetworkHealthSummary from SensorHealthMonitor (Phase 2).
        advisories: Optional Dict of MaintenanceAdvisory from MaintenanceAdvisor (Phase 3).
        maintenance_summary: Optional NetworkMaintenanceSummary from MaintenanceAdvisor (Phase 3).
        investigation_summary: Optional compiled report dictionary from build_investigation_summary (Phase 4).
    """
    render_header(st_ctx)

    tabs = st_ctx.tabs([
        "Overview",
        "Sensor Monitoring",
        "Network Topology",
        "Fault Injection",
        "Anomaly Detection",
        "Root-Cause Localization",
        "Explainable Investigation",
        "Sensor Health Center",
        "Predictive Maintenance Center",
        "Executive Reports",
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
        render_explainable_investigation(st_ctx, investigation_report, rankings)

    with tabs[7]:
        render_sensor_health_center(st_ctx, health_reports, network_summary=network_health)

    with tabs[8]:
        render_predictive_maintenance_center(st_ctx, advisories, maintenance_summary=maintenance_summary, health_reports=health_reports)

    with tabs[9]:
        render_executive_reports_tab(st_ctx, investigation_summary, readings=readings, scores=scores, rankings=rankings, benchmark=benchmark)

    with tabs[10]:
        build_benchmark_evaluation(st_ctx, benchmark)

    with tabs[11]:
        build_export_section(st_ctx, readings, scores, rankings)


# -----------------------------------------------------------------------------
# Live IoT Mode Layout & Sidebar Controls
# -----------------------------------------------------------------------------

def sidebar_mode_selector(st_ctx: Any = st) -> str:
    """Render top-level operating mode selector radio."""
    st_ctx.sidebar.subheader("System Operating Mode")
    mode = st_ctx.sidebar.radio(
        "Select Operating Mode",
        options=["Simulation Mode", "Live IoT Mode"],
        index=0,
        help="Switch between synthetic simulation experiments and real-time MQTT telemetry streaming.",
    )
    st_ctx.sidebar.markdown("---")
    return str(mode)


def sidebar_live_iot_controls(
    st_ctx: Any,
    mqtt_mgr: Any,
    gw_sim: Any,
    telem_mgr: Any,
) -> None:
    """Render sidebar configuration panels for Live IoT Mode."""
    st_ctx.sidebar.header("MQTT Broker Settings")

    broker_host = st_ctx.sidebar.text_input(
        "Broker Host",
        value=mqtt_mgr.config.host,
        help="MQTT broker IP or hostname (e.g. localhost or broker.hivemq.com).",
    )
    broker_port = st_ctx.sidebar.number_input(
        "Broker Port",
        min_value=1,
        max_value=65535,
        value=int(mqtt_mgr.config.port),
        step=1,
        help="Standard MQTT port (default 1883).",
    )
    broker_topic = st_ctx.sidebar.text_input(
        "Subscribe Topic",
        value=mqtt_mgr.config.topic,
        help="Topic filter for telemetry subscription (e.g. faultmesh/sensors/#).",
    )

    mqtt_mgr.config.host = broker_host.strip()
    mqtt_mgr.config.port = int(broker_port)
    mqtt_mgr.config.topic = broker_topic.strip()

    is_tls = bool(mqtt_mgr.config.use_tls or int(broker_port) == 8883)
    use_tls_toggle = st_ctx.sidebar.checkbox(
        "Enable TLS / MQTTS (Port 8883)",
        value=is_tls,
        help="Enable secure TLS certificate and hostname verification for encrypted broker connectivity.",
    )
    mqtt_mgr.config.use_tls = use_tls_toggle

    # Authentication credentials expander (preserves cloud secrets, masks passwords)
    with st_ctx.sidebar.expander("Authentication & Credentials", expanded=False):
        current_user = mqtt_mgr.config.username or ""
        input_user = st_ctx.sidebar.text_input(
            "Subscriber Username",
            value=current_user,
            help="Username for broker authentication.",
            key="sb_mqtt_user",
        )
        input_pwd = st_ctx.sidebar.text_input(
            "Subscriber Password",
            type="password",
            value="",
            placeholder="••••••••" if mqtt_mgr.config.password else "Enter password",
            help="Password for broker authentication (leave blank to keep configured secret).",
            key="sb_mqtt_pwd",
        )
        if input_user != current_user:
            mqtt_mgr.config.username = input_user.strip() if input_user.strip() else None
        if input_pwd.strip():
            mqtt_mgr.config.password = input_pwd.strip()

    # Connection buttons
    status = mqtt_mgr.get_status()
    is_connected = status.get("is_connected", False)

    col1, col2 = st_ctx.sidebar.columns(2)
    with col1:
        if col1.button("Connect", use_container_width=True, type="primary", disabled=is_connected):
            ok, msg = mqtt_mgr.connect()
            if ok:
                st_ctx.sidebar.success(msg)
            else:
                st_ctx.sidebar.error(msg)
            st_ctx.rerun()

    with col2:
        if col2.button("Disconnect", use_container_width=True, disabled=not is_connected):
            ok, msg = mqtt_mgr.disconnect()
            st_ctx.sidebar.info(msg)
            st_ctx.rerun()

    if is_connected:
        if st_ctx.sidebar.button("Update Subscription", use_container_width=True, help="Update MQTT topic subscription filter."):
            ok, msg = mqtt_mgr.subscribe(broker_topic.strip())
            if ok:
                st_ctx.sidebar.success(msg)
            else:
                st_ctx.sidebar.error(msg)
            st_ctx.rerun()

    st_ctx.sidebar.markdown("---")

    # Software Gateway Simulator Panel
    st_ctx.sidebar.header("Software Gateway Simulator")
    st_ctx.sidebar.caption("Software simulation publisher for local test and demo. Not connected to physical IoT hardware.")

    gw_status = gw_sim.get_status()
    is_gw_running = gw_status.get("is_running", False)

    scenario = st_ctx.sidebar.selectbox(
        "Gateway Scenario",
        options=["normal", "drift", "spike", "stuck"],
        index=["normal", "drift", "spike", "stuck"].index(gw_status.get("active_scenario", "normal")),
        help="Select fault scenario to inject into the live publisher stream.",
    )

    mag_ratio = st_ctx.sidebar.slider(
        "Fault Magnitude Ratio",
        min_value=0.1,
        max_value=1.5,
        value=0.5,
        step=0.05,
        disabled=(scenario == "normal"),
    )

    interval = st_ctx.sidebar.slider(
        "Publish Interval (seconds)",
        min_value=0.2,
        max_value=3.0,
        value=float(gw_status.get("publish_interval", 1.0)),
        step=0.1,
        help="Time between consecutive multi-sensor telemetry packets.",
    )

    gw_sim.set_scenario(
        scenario=scenario,
        fault_sensor=gw_status.get("fault_sensor", "S3"),
        magnitude_ratio=mag_ratio,
        fault_start_step=20,
        spike_duration=10,
    )
    gw_sim.publish_interval = float(interval)
    gw_sim.broker_host = mqtt_mgr.config.host
    gw_sim.broker_port = mqtt_mgr.config.port
    gw_sim.use_tls = bool(mqtt_mgr.config.use_tls or mqtt_mgr.config.port == 8883)

    col_gw1, col_gw2 = st_ctx.sidebar.columns(2)
    with col_gw1:
        if st_ctx.button("Start Gateway", use_container_width=True, type="primary", disabled=is_gw_running):
            ok, msg = gw_sim.start()
            if ok:
                st_ctx.sidebar.success(msg)
            else:
                st_ctx.sidebar.error(msg)
            st_ctx.rerun()

    with col_gw2:
        if st_ctx.button("Stop Gateway", use_container_width=True, disabled=not is_gw_running):
            ok, msg = gw_sim.stop()
            st_ctx.sidebar.info(msg)
            st_ctx.rerun()

    st_ctx.sidebar.markdown("---")

    # Telemetry Refresh Controls
    st_ctx.sidebar.subheader("Telemetry Streaming Controls")
    auto_refresh = st_ctx.sidebar.toggle(
        "Auto-Refresh Stream (1s)",
        value=True,
        help="Continuously sync telemetry and refresh diagnostic graphs every 1 second.",
    )
    st_ctx.session_state["auto_refresh_telemetry"] = auto_refresh

    if st_ctx.sidebar.button("Refresh Telemetry", use_container_width=True):
        st_ctx.rerun()


def _build_live_iot_view(
    st_ctx: Any,
    mqtt_mgr: Any,
    gw_sim: Any,
    telem_mgr: Any,
    benchmark: Optional[Dict[str, Any]] = None,
) -> None:
    """Internal builder for complete Live IoT dashboard view."""
    # 0. Drain thread-safe MQTT message queue from the main thread
    mqtt_mgr.drain_message_queue(max_messages=250)

    st_ctx.title("FaultMesh AI -- Live IoT Telemetry Mode")
    st_ctx.markdown(
        "**Real-time industrial IoT sensor telemetry ingestion via MQTT.** "
        "Continuously evaluates multi-sensor streams, enforces baseline calibration, "
        "and isolates root-cause anomalies across physical mesh connections."
    )
    broker_endpoint = f"`{mqtt_mgr.config.host}:{mqtt_mgr.config.port}`"
    tls_badge = "🔒 **TLS / MQTTS (Encrypted)**" if (mqtt_mgr.config.use_tls or mqtt_mgr.config.port == 8883) else "🔓 **Plaintext TCP**"
    auth_badge = f"👤 User: `{mqtt_mgr.config.username}`" if mqtt_mgr.config.username else "👤 Anonymous"
    st_ctx.caption(f"Broker Endpoint: {broker_endpoint} &nbsp;|&nbsp; Security: {tls_badge} &nbsp;|&nbsp; Auth: {auth_badge}")
    st_ctx.info(
        "**Notice:** Industrial decision-support telemetry monitor. "
        "Read-only ingestion mode — does not issue actuator commands or override plant safety interlocks."
    )

    # 1. Communication & Validation Statistics
    stats = mqtt_mgr.validator.get_stats()
    mqtt_status = mqtt_mgr.get_status()
    render_live_comm_stats(st_ctx, stats, mqtt_status)

    st_ctx.markdown("---")

    # 2. Baseline Calibration Panel
    render_calibration_panel(st_ctx, telem_mgr)

    st_ctx.markdown("---")

    # 3. Live 5-Sensor KPI Metric Cards
    latest_readings = telem_mgr.get_latest_readings()
    render_live_sensor_cards(st_ctx, latest_readings)

    # 4. Execute AI Analysis if Calibrated
    analysis_res = telem_mgr.process_live_analysis(benchmark=benchmark)
    is_calibrated = (analysis_res.get("status") == "success")

    # Render Live Diagnostic Transparency Table
    sensor_diagnostics = analysis_res.get("sensor_diagnostics") if is_calibrated else telem_mgr.get_sensor_diagnostics()
    render_live_diagnostic_transparency_table(st_ctx, sensor_diagnostics)

    readings_df = telem_mgr.get_synchronized_dataframe()
    scores_df = analysis_res.get("scores") if is_calibrated else None
    onsets = analysis_res.get("onsets") if is_calibrated else None
    rankings = analysis_res.get("rankings") if is_calibrated else None
    inv_report = analysis_res.get("investigation_report") if is_calibrated else None
    health_reports = analysis_res.get("health_reports") if is_calibrated else None
    net_health = analysis_res.get("network_health") if is_calibrated else None
    advisories = analysis_res.get("advisories") if is_calibrated else None
    maint_summary = analysis_res.get("maintenance_summary") if is_calibrated else None
    exec_summary = analysis_res.get("executive_summary") if is_calibrated else None

    from faultmesh.graph import SensorNetworkGraph
    network_graph = SensorNetworkGraph()

    tabs = st_ctx.tabs([
        "Live Sensor Streams",
        "Network Topology",
        "Anomaly Detection",
        "Root-Cause Localization",
        "Explainable Investigation",
        "Sensor Health Center",
        "Predictive Maintenance Center",
        "Executive Reports",
        "MQTT & Ingestion Logs",
        "Benchmark Reference",
    ])

    # Tab 0: Live Sensor Streams
    with tabs[0]:
        st_ctx.subheader("Real-Time Multi-Sensor Telemetry Streams")
        if readings_df is not None and not readings_df.empty:
            render_sensor_timeseries(st_ctx, readings_df, scores_df)
            with st_ctx.expander("Recent Synchronized Telemetry Points (Last 10)"):
                st_ctx.dataframe(readings_df.tail(10), use_container_width=True, hide_index=True)
        else:
            st_ctx.warning("No synchronized telemetry points received yet. Connect to broker and start data stream.")

    # Tab 1: Network Topology
    with tabs[1]:
        build_network_topology(st_ctx, network_graph, rankings, onsets)

    # Tab 2: Anomaly Detection
    with tabs[2]:
        build_anomaly_detection(
            st_ctx=st_ctx,
            scores=scores_df,
            onsets=onsets,
            latest_readings=latest_readings,
            sensor_diagnostics=sensor_diagnostics,
            readings_df=readings_df,
        )

    # Tab 3: Root-Cause Localization
    with tabs[3]:
        diag_status = analysis_res.get("fault_diagnosis_status", "NORMAL")
        primary_suspect = analysis_res.get("primary_suspect")
        build_localization(
            st_ctx,
            rankings,
            onsets=onsets,
            diagnosis_status=diag_status,
            primary_suspect=primary_suspect,
        )

    # Tab 4: Explainable Investigation
    with tabs[4]:
        render_explainable_investigation(st_ctx, inv_report, rankings)

    # Tab 5: Sensor Health Center
    with tabs[5]:
        if is_calibrated:
            render_sensor_health_center(st_ctx, health_reports, network_summary=net_health)
        else:
            st_ctx.info("Sensor Health Center Suspended: Calibration required.")

    # Tab 6: Predictive Maintenance Center
    with tabs[6]:
        if is_calibrated:
            render_predictive_maintenance_center(st_ctx, advisories, maintenance_summary=maint_summary, health_reports=health_reports)
        else:
            st_ctx.info("Predictive Maintenance Suspended: Calibration required.")

    # Tab 7: Executive Reports
    with tabs[7]:
        if is_calibrated and exec_summary is not None:
            render_executive_reports_tab(
                st_ctx,
                exec_summary,
                readings=readings_df,
                scores=scores_df,
                rankings=rankings,
                benchmark=benchmark,
            )
        else:
            st_ctx.info("Executive Reporting Suspended: Complete baseline calibration to generate reports.")

    # Tab 8: MQTT & Ingestion Logs
    with tabs[8]:
        st_ctx.subheader("MQTT Ingestion Telemetry & Diagnostic Logs")

        # Operational Distinction Notice
        st_ctx.info(
            "ℹ️ **Operational Diagnostic Notice:**\n"
            "- **Broker Connection:** Validates network TCP socket connectivity with the MQTT broker.\n"
            "- **Subscription Acknowledgement:** Confirms the broker accepted topic filter registration.\n"
            "- **Message Reception:** Callbacks trigger exclusively when external publishers transmit packets to this topic. "
            "If callbacks remain zero, verify the publishing topic and broker host/port."
        )

        # High-level Diagnostic KPI Cards
        d_col1, d_col2, d_col3, d_col4 = st_ctx.columns(4)
        d_col1.metric(
            label="Connection Status",
            value=mqtt_status.get("connection_state", "DISCONNECTED"),
            delta=f"Port: {mqtt_status.get('broker_port', 1883)}",
        )
        d_col2.metric(
            label="Subscription Status",
            value=mqtt_status.get("subscription_status", "Inactive"),
            delta=f"Topic: {mqtt_status.get('topic', 'N/A')}",
        )
        d_col3.metric(
            label="Callback / Queue Sync",
            value=f"{mqtt_status.get('callbacks_received', 0):,} recvd",
            delta=f"{mqtt_status.get('messages_drained', 0):,} drained",
        )
        d_col4.metric(
            label="Queue Buffer / Pending",
            value=f"{mqtt_status.get('pending_queue_size', 0)} buffered",
            delta=f"Valid: {stats['valid_count']} | Rej: {stats['rejected_count']}",
        )

        st_ctx.divider()

        # Telemetry Ingestion Diagnostics Overview Table
        diag_data = {
            "Diagnostic Parameter": [
                "MQTT Client Instance ID",
                "Client ID (Paho)",
                "Paho Client Pointer",
                "Paho Network Loop Status",
                "Broker Host & Port",
                "TLS Encryption Status",
                "CA Certificate Verification",
                "Authentication Status",
                "Connection State",
                "Subscription Status",
                "Configured Topic",
                "Last Connected Timestamp",
                "Last Subscribed Timestamp",
                "Thread-Safe Queue Identity",
                "Pending Queue Size",
                "Total Background Callbacks",
                "Total Main-Thread Drained",
                "Valid Sensor Readings",
                "Rejected/Corrupted Packets",
                "Ingestion Error Rate",
                "Last Received Topic",
                "Last Received Timestamp",
                "Latest Validation Error",
                "Latest Queue-Drain Error",
            ],
            "Value": [
                mqtt_status.get("client_instance_id", "N/A"),
                mqtt_status.get("client_id", "N/A"),
                mqtt_status.get("paho_client_ptr", "N/A"),
                mqtt_status.get("network_loop_status", "Stopped"),
                f"{mqtt_status.get('broker_host', 'localhost')}:{mqtt_status.get('broker_port', 1883)}",
                "Enabled (TLS / MQTTS)" if mqtt_status.get("use_tls") else "Disabled (Plaintext TCP)",
                "Strict Verification (CERT_REQUIRED, check_hostname=True)" if mqtt_status.get("use_tls") else "N/A (Plaintext)",
                f"Authenticated ({mqtt_status.get('username')})" if mqtt_status.get("has_auth") else "Anonymous",
                mqtt_status.get("connection_state", "DISCONNECTED"),
                mqtt_status.get("subscription_status", "Inactive"),
                mqtt_status.get("topic", "N/A"),
                mqtt_status.get("last_connected_at") or "Not yet connected",
                mqtt_status.get("last_subscribed_at") or "Not yet subscribed",
                mqtt_status.get("queue_identity", "N/A"),
                f"{mqtt_status.get('pending_queue_size', 0)} packets",
                f"{mqtt_status.get('callbacks_received', 0):,} packets",
                f"{mqtt_status.get('messages_drained', 0):,} packets",
                f"{stats['valid_count']:,} readings",
                f"{stats['rejected_count']:,} readings",
                f"{stats['error_rate_pct']:.2f}%",
                mqtt_status.get("last_received_topic") or "None received yet",
                mqtt_status.get("last_message_at") or "None received yet",
                stats.get("last_error") or "None (Clean stream)",
                mqtt_status.get("last_drain_error") or "None (No drain errors)",
            ],
        }
        st_ctx.markdown("#### Real-Time Ingestion Diagnostic Summary")
        st_ctx.dataframe(pd.DataFrame(diag_data), use_container_width=True, hide_index=True)

        col_log1, col_log2 = st_ctx.columns(2)
        with col_log1:
            st_ctx.markdown("#### Message Validation Breakdown")
            if stats.get("last_error"):
                st_ctx.error(f"**Latest Validation Error:** {stats['last_error']}")
            if mqtt_status.get("last_drain_error"):
                st_ctx.error(f"**Latest Queue-Drain Error:** {mqtt_status['last_drain_error']}")
            if stats["errors_by_type"]:
                st_ctx.write("**Errors by Type:**")
                st_ctx.json(stats["errors_by_type"])
            st_ctx.write("**Packets per Sensor:**")
            st_ctx.json(stats["counts_by_sensor"])

        with col_log2:
            st_ctx.markdown("#### Recent Ingestion Event Log")
            logs = mqtt_status.get("recent_logs", [])
            if logs:
                st_ctx.text_area("Recent Event Logs", value="\n".join(reversed(logs)), height=260)
            else:
                st_ctx.caption("No log events recorded yet.")

        # Rejection Diagnostic Table
        rejections = stats.get("recent_rejections", [])
        if rejections:
            st_ctx.markdown("#### Recent Rejected Packets Diagnostic Audit (Last 25)")
            st_ctx.dataframe(pd.DataFrame(rejections), use_container_width=True, hide_index=True)

    # Tab 9: Benchmark Reference
    with tabs[9]:
        build_benchmark_evaluation(st_ctx, benchmark)


@st.fragment(run_every="1s")
def _live_telemetry_streaming_fragment(
    st_ctx: Any,
    mqtt_mgr: Any,
    gw_sim: Any,
    telem_mgr: Any,
    benchmark: Optional[Dict[str, Any]] = None,
) -> None:
    """Auto-refreshing Live IoT telemetry view fragment."""
    _build_live_iot_view(st_ctx, mqtt_mgr, gw_sim, telem_mgr, benchmark)


def _live_telemetry_static_render(
    st_ctx: Any,
    mqtt_mgr: Any,
    gw_sim: Any,
    telem_mgr: Any,
    benchmark: Optional[Dict[str, Any]] = None,
) -> None:
    """Static Live IoT telemetry view rendered without auto-refresh."""
    _build_live_iot_view(st_ctx, mqtt_mgr, gw_sim, telem_mgr, benchmark)


def render_live_iot_dashboard(
    st_ctx: Any,
    mqtt_mgr: Any,
    gw_sim: Any,
    telem_mgr: Any,
    benchmark: Optional[Dict[str, Any]] = None,
) -> None:
    """Assemble complete Live IoT Mode monitoring and diagnostic interface.

    Dispatches to 1-second auto-refresh streaming fragment or static render
    based on sidebar user preference.
    """
    auto_refresh = st.session_state.get("auto_refresh_telemetry", True)
    if auto_refresh:
        _live_telemetry_streaming_fragment(st_ctx, mqtt_mgr, gw_sim, telem_mgr, benchmark)
    else:
        _live_telemetry_static_render(st_ctx, mqtt_mgr, gw_sim, telem_mgr, benchmark)

