"""Streamlit Dashboard for FaultMesh AI.

Responsibility:
    Provide an interactive web dashboard for real-time sensor network monitoring,
    fault injection simulation, anomaly score visualization, network topology rendering,
    and root-cause localization ranking.

Usage:
    streamlit run app.py
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

# Component imports
from faultmesh.pipeline import FaultMeshPipeline
from faultmesh.graph import SensorNetworkGraph
from faultmesh.simulator import DEFAULT_BASELINES


def init_page_config() -> None:
    """Set up Streamlit page title, layout, and meta settings."""
    st.set_page_config(
        page_title="FaultMesh AI — IoT Fault Localization",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> Dict[str, Any]:
    """Render interactive simulation controls in the sidebar.

    Returns:
        Dictionary of user-selected configuration parameters.
    """
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("Configure IoT mesh simulation and fault injection parameters.")

    target_sensor = st.sidebar.selectbox(
        "Fault Injection Sensor",
        options=["S1", "S2", "S3", "S4", "S5"],
        index=0,
        help="Target sensor to inject physical or electrical fault.",
    )

    fault_type = st.sidebar.selectbox(
        "Fault Type",
        options=["drift", "spike", "stuck"],
        index=0,
        help="Type of fault profile to introduce.",
    )

    total_steps = st.sidebar.slider(
        "Total Time Steps",
        min_value=120,
        max_value=500,
        value=200,
        step=20,
    )

    fault_start = st.sidebar.slider(
        "Fault Start Timestep",
        min_value=50,
        max_value=total_steps - 20,
        value=100,
        step=10,
    )

    run_button = st.sidebar.button("🚀 Run Simulation & Localization", use_container_width=True)

    return {
        "target_sensor": target_sensor,
        "fault_type": fault_type,
        "total_steps": total_steps,
        "fault_start": fault_start,
        "run_button": run_button,
    }


def main() -> None:
    """Main dashboard application entry point."""
    init_page_config()

    st.title("🛰️ FaultMesh AI")
    st.subheader("Intelligent Cooperative IoT Fault Localization System")
    st.markdown(
        """
        Welcome to **FaultMesh AI**. This system monitors multi-sensor IoT network streams,
        filters noise, detects multivariate anomalies via Isolation Forests, and determines
        the root cause of cascading failures using topological graph reasoning.
        """
    )

    config = render_sidebar()

    # Layout columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("👈 Use the sidebar to configure simulation parameters and run fault diagnosis.")
        st.subheader("Sensor Time Series Readings")
        # TODO: Plot raw and coupled sensor readings using Plotly
        st.caption("Plots will display here once simulation is executed.")

    with col2:
        st.subheader("Sensor Network Topology")
        # TODO: Render NetworkX sensor mesh using Plotly graph / pyvis
        st.caption("Network nodes: S1..S5 with coupling edges.")

        st.subheader("Root-Cause Localization Ranking")
        # TODO: Display ranked candidate table with composite scores
        st.caption("Localization results will appear after running simulation.")


if __name__ == "__main__":
    main()
