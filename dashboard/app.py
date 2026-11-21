"""Main entry point for the FaultMesh AI Streamlit dashboard.

Responsibility:
    Initializes Streamlit page configuration, manages user interaction and
    pipeline execution in the sidebar, maintains atomic session state, and
    renders the eight-section layout without duplicating backend logic.
"""

from pathlib import Path
from dataclasses import asdict
import streamlit as st

from faultmesh.graph import SensorNetworkGraph
from faultmesh.ingestion import (
    LiveTelemetryManager,
    DataValidator,
    MQTTClientManager,
    MQTTConnectionConfig,
    GatewaySimulator,
)
from dashboard.utils import (
    SimConfig,
    run_simulation,
    run_detection,
    run_localization,
    load_benchmark,
    run_explainability,
    run_health_assessment,
    run_maintenance_advisories,
    build_investigation_summary,
    get_global_live_iot_subsystem,
)
from dashboard.layout import (
    sidebar_mode_selector,
    sidebar_controls,
    sidebar_live_iot_controls,
    render_dashboard_layout,
    render_live_iot_dashboard,
)

# -----------------------------------------------------------------------------
# 1. Streamlit Configuration (Must be the first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FaultMesh AI -- Intelligent IoT Fault Localization",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safe stylesheet loading using pathlib
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.is_file():
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as err:
        st.warning(f"Could not load custom stylesheet: {err}")

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "simulation_executed" not in st.session_state:
    st.session_state.simulation_executed = False
    st.session_state.active_cfg = None
    st.session_state.readings = None
    st.session_state.scores = None
    st.session_state.onsets = None
    st.session_state.rankings = None
    st.session_state.investigation_report = None
    st.session_state.health_reports = None
    st.session_state.network_health = None
    st.session_state.maintenance_advisories = None
    st.session_state.maintenance_summary = None
    st.session_state.investigation_summary = None

# Independent and failure-safe benchmark loading
if "benchmark" not in st.session_state:
    try:
        st.session_state.benchmark = load_benchmark()
        if st.session_state.benchmark is None:
            st.warning("Benchmark results not found at data/evaluation_results.csv. Run 'python evaluate.py' to generate.")
    except Exception as exc:
        st.session_state.benchmark = None
        st.warning(f"Benchmark data could not be loaded: {exc}")

# Static network topology graph instance
if "network_graph" not in st.session_state or st.session_state.network_graph is None:
    st.session_state.network_graph = SensorNetworkGraph()

# Live IoT Manager Instances (Singleton lifecycle managed via st.cache_resource)
live_subsystem = get_global_live_iot_subsystem()
st.session_state.validator = live_subsystem["validator"]
st.session_state.telem_mgr = live_subsystem["telem_mgr"]
st.session_state.mqtt_mgr = live_subsystem["mqtt_mgr"]
st.session_state.gw_sim = live_subsystem["gw_sim"]

# -----------------------------------------------------------------------------
# 3. Operating Mode Selection
# -----------------------------------------------------------------------------
operating_mode = sidebar_mode_selector(st)

if operating_mode == "Simulation Mode":
    # -------------------------------------------------------------------------
    # Simulation Controls & Execution
    # -------------------------------------------------------------------------
    current_cfg: SimConfig = sidebar_controls(st)
    run_clicked = st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True,
        help="Execute multi-sensor simulation, anomaly scoring, and graph-guided localization.",
    )

    if run_clicked:
        try:
            with st.spinner("Executing simulation and diagnostic pipeline..."):
                # 1. Run physical simulation with Gaussian noise and neighbor coupling
                readings, fault_dict = run_simulation(current_cfg)

                # 2. Feature processing and Isolation Forest detection (trained on t=0..99)
                scores, onsets = run_detection(readings)

                # 3. Graph-guided root-cause localization without ground-truth leakage
                rankings = run_localization(scores, onsets)

                # 4. Sensor network graph
                graph = SensorNetworkGraph()

                # 5. Explainable Fault Investigation Engine (Phase 1)
                inv_rep = run_explainability(scores, onsets, rankings, graph=graph)

                # 6. Sensor Health Monitoring Engine (Phase 2)
                health_reps, network_health = run_health_assessment(readings, scores, onsets=onsets)

                # 7. Predictive Maintenance Recommendation Engine (Phase 3)
                maint_advisories, maint_summary = run_maintenance_advisories(
                    health_reports=health_reps,
                    rankings=rankings,
                    investigation_report=inv_rep,
                    scores_df=scores,
                    onsets=onsets,
                )

                # 8. Non-blocking benchmark reload: failure must not break simulation pipeline
                try:
                    bmk = load_benchmark()
                except Exception:
                    bmk = getattr(st.session_state, "benchmark", None)

                # 9. Executive Investigation Summary (Phase 4)
                inv_summary = build_investigation_summary(
                    cfg_info=current_cfg.__dict__ if hasattr(current_cfg, "__dict__") else {},
                    readings_df=readings,
                    scores_df=scores,
                    onsets=onsets,
                    rankings=rankings,
                    investigation_report=inv_rep,
                    health_reports=health_reps,
                    advisories=maint_advisories,
                    network_health=network_health,
                    maintenance_summary=maint_summary,
                    benchmark=bmk,
                )

                # Persist report to SQLite storage for historical audit
                try:
                    from faultmesh.storage import save_report
                    save_report(inv_summary)
                except Exception:
                    pass

                # Atomically update session state on complete pipeline success
                st.session_state.active_cfg = current_cfg
                st.session_state.readings = readings
                st.session_state.scores = scores
                st.session_state.onsets = onsets
                st.session_state.rankings = rankings
                st.session_state.network_graph = graph
                st.session_state.investigation_report = inv_rep
                st.session_state.health_reports = health_reps
                st.session_state.network_health = network_health
                st.session_state.maintenance_advisories = maint_advisories
                st.session_state.maintenance_summary = maint_summary
                st.session_state.investigation_summary = inv_summary
                if bmk is not None:
                    st.session_state.benchmark = bmk

                st.session_state.simulation_executed = True
                st.sidebar.success("Pipeline executed successfully.")
        except Exception as exc:
            st.error(f"Pipeline execution failed: {type(exc).__name__}: {exc}. Previous results have been preserved.")

    # Render Simulation Dashboard Layout
    if not st.session_state.simulation_executed:
        st.info("Ready to Simulate: Configure parameters in the sidebar and click Run Simulation.")
        render_dashboard_layout(
            st_ctx=st,
            cfg=current_cfg,
            readings=None,
            scores=None,
            onsets=None,
            rankings=None,
            network_graph=st.session_state.network_graph,
            benchmark=st.session_state.benchmark,
            investigation_report=None,
            health_reports=None,
            network_health=None,
            advisories=None,
            maintenance_summary=None,
            investigation_summary=None,
        )
    else:
        render_dashboard_layout(
            st_ctx=st,
            cfg=st.session_state.active_cfg,
            readings=st.session_state.readings,
            scores=st.session_state.scores,
            onsets=st.session_state.onsets,
            rankings=st.session_state.rankings,
            network_graph=st.session_state.network_graph,
            benchmark=st.session_state.benchmark,
            investigation_report=st.session_state.investigation_report,
            health_reports=st.session_state.health_reports,
            network_health=st.session_state.network_health,
            advisories=st.session_state.maintenance_advisories,
            maintenance_summary=st.session_state.maintenance_summary,
            investigation_summary=st.session_state.investigation_summary,
        )

else:
    # -------------------------------------------------------------------------
    # Live IoT Mode
    # -------------------------------------------------------------------------
    sidebar_live_iot_controls(
        st_ctx=st,
        mqtt_mgr=st.session_state.mqtt_mgr,
        gw_sim=st.session_state.gw_sim,
        telem_mgr=st.session_state.telem_mgr,
    )
    render_live_iot_dashboard(
        st_ctx=st,
        mqtt_mgr=st.session_state.mqtt_mgr,
        gw_sim=st.session_state.gw_sim,
        telem_mgr=st.session_state.telem_mgr,
        benchmark=st.session_state.benchmark,
    )

