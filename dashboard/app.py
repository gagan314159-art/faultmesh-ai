"""Main entry point for the FaultMesh AI Streamlit dashboard.

Responsibility:
    Initializes Streamlit page configuration, manages user interaction and
    pipeline execution in the sidebar, maintains atomic session state, and
    renders the eight-section layout without duplicating backend logic.
"""

from pathlib import Path
import streamlit as st

from faultmesh.graph import SensorNetworkGraph
from dashboard.utils import (
    SimConfig,
    run_simulation,
    run_detection,
    run_localization,
    load_benchmark,
)
from dashboard.layout import (
    sidebar_controls,
    render_dashboard_layout,
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
else:
    st.warning("Dashboard stylesheet (styles.css) not found -- continuing with default Streamlit theme.")

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

# -----------------------------------------------------------------------------
# 3. Sidebar Simulation Controls & Execution
# -----------------------------------------------------------------------------
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

            # 5. Non-blocking benchmark reload: failure must not break simulation pipeline
            try:
                bmk = load_benchmark()
            except Exception:
                bmk = None

            # Atomically update session state on complete pipeline success
            st.session_state.active_cfg = current_cfg
            st.session_state.readings = readings
            st.session_state.scores = scores
            st.session_state.onsets = onsets
            st.session_state.rankings = rankings
            st.session_state.network_graph = graph
            if bmk is not None:
                st.session_state.benchmark = bmk

            st.session_state.simulation_executed = True
            st.sidebar.success("Pipeline executed successfully.")
    except Exception as exc:
        st.error(f"Pipeline execution failed: {exc}. Previous results have been preserved.")

# -----------------------------------------------------------------------------
# 4. Render Dashboard Layout
# -----------------------------------------------------------------------------
if not st.session_state.simulation_executed:
    st.info("Ready to Simulate: Configure parameters in the sidebar and click Run Simulation.")
    # Render layout with unpopulated placeholders, enabling topology and benchmark inspection
    render_dashboard_layout(
        st_ctx=st,
        cfg=current_cfg,
        readings=None,
        scores=None,
        onsets=None,
        rankings=None,
        network_graph=st.session_state.network_graph,
        benchmark=st.session_state.benchmark,
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
    )
