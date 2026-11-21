"""Reusable Streamlit and Plotly dashboard components for FaultMesh AI.

Responsibility:
    Provides modular, independent rendering components for KPI cards, sensor
    time-series, anomaly scores, network topology, localization ranking tables,
    benchmark performance summaries, and report download controls.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# Dark theme palette aligned with FaultMesh AI industrial design
THEME_BG = "rgba(11, 25, 44, 0.9)"
THEME_PLOT_BG = "rgba(15, 32, 67, 0.9)"
ACCENT_CYAN = "#00e5ff"
ACCENT_BLUE = "#0099ff"
ACCENT_AMBER = "#ffaa00"
ACCENT_RED = "#ff3366"
ACCENT_GREEN = "#00e676"


def render_kpi_cards(
    st: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    onsets: Optional[Dict[str, Optional[int]]],
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Render high-level KPI cards for monitoring overview.

    Displays sensor count, total timesteps/observations, alarmed sensor count,
    and highest-ranked suspect. Accurately reports zero when data is missing.

    Args:
        st: Streamlit module or container.
        readings: Raw sensor readings DataFrame.
        scores: Anomaly scores DataFrame.
        onsets: Detection onset timestep dictionary per sensor.
        rankings: Ranked suspect tuples from FaultLocalizer.
    """
    col1, col2, col3, col4 = st.columns(4)

    has_readings = readings is not None and not readings.empty
    sensor_cols = [c for c in readings.columns if c not in ("time", "timestep") and not c.startswith("_")] if has_readings else []
    sensor_count = len(sensor_cols)

    # 1. Monitored Sensors count
    if sensor_count > 0:
        col1.metric(
            label="Monitored Sensors",
            value=f"{sensor_count} Sensors",
            delta="Mesh Active",
        )
    else:
        col1.metric(
            label="Monitored Sensors",
            value="0 Sensors",
            delta="No Active Sensors",
        )

    # 2. Total timesteps and observations
    total_steps = len(readings) if has_readings else 0
    total_data_points = total_steps * sensor_count

    col2.metric(
        label="Total Time Steps",
        value=f"{total_steps:,}",
        delta=f"{total_data_points:,} readings" if total_data_points > 0 else "No data",
    )

    # 3. Alarmed sensors
    alarmed_count = 0
    if onsets and sensor_count > 0:
        alarmed_count = sum(1 for s in sensor_cols if onsets.get(s) is not None)

    if sensor_count > 0:
        delta_label = "System Abnormal" if alarmed_count > 0 else "System Nominal"
        col3.metric(
            label="Sensors with Onset",
            value=f"{alarmed_count} / {sensor_count}",
            delta=delta_label,
            delta_color="inverse" if alarmed_count > 0 else "normal",
        )
    else:
        col3.metric(
            label="Sensors with Onset",
            value="0 / 0",
            delta="No Data",
        )

    # 4. Top suspect root cause
    if rankings and len(rankings) > 0 and sensor_count > 0:
        top_suspect, top_score, _ = rankings[0]
        col4.metric(
            label="Top Root Cause Suspect",
            value=f"Sensor {top_suspect}",
            delta=f"Score: {top_score:.3f}",
            delta_color="inverse",
        )
    else:
        col4.metric(
            label="Top Root Cause Suspect",
            value="None",
            delta="Nominal",
        )


def render_sensor_timeseries(
    st: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    selected_sensors: Optional[List[str]] = None,
) -> None:
    """Plot multi-sensor readings and calibrated anomaly scores using Plotly.

    Args:
        st: Streamlit module or container.
        readings: Sensor readings DataFrame.
        scores: Calibrated anomaly scores DataFrame.
        selected_sensors: List of sensor IDs to display. Defaults to all available sensors.
    """
    if readings is None or readings.empty:
        st.info("No sensor reading data available.")
        return

    available_sensors = [c for c in readings.columns if c not in ("time", "timestep") and not c.startswith("_")]
    if not available_sensors:
        st.warning("No valid sensor columns found in dataset.")
        return

    # Filter by user selection or default to all available sensors
    if selected_sensors:
        active_sensors = [s for s in selected_sensors if s in available_sensors]
        if not active_sensors:
            active_sensors = available_sensors
    else:
        active_sensors = available_sensors

    time_col = "time" if "time" in readings.columns else ("timestep" if "timestep" in readings.columns else None)
    x_axis = readings[time_col] if time_col else readings.index

    # 1. Raw Time-Series Chart
    fig_readings = go.Figure()
    palette = [ACCENT_CYAN, ACCENT_BLUE, "#9c27b0", "#ff9800", ACCENT_GREEN, "#e91e63"]

    for idx, sensor in enumerate(active_sensors):
        color = palette[idx % len(palette)]
        fig_readings.add_trace(go.Scatter(
            x=x_axis,
            y=readings[sensor],
            mode="lines",
            name=f"{sensor} Reading",
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{sensor}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
        ))

    fig_readings.update_layout(
        title="Physical Sensor Time-Series Readings",
        xaxis_title="Simulation Time Step",
        yaxis_title="Reading Value",
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_PLOT_BG,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_readings, use_container_width=True)

    # 2. Anomaly Scores Chart
    if scores is not None and not scores.empty:
        fig_scores = go.Figure()
        scores_x = scores[time_col] if time_col and time_col in scores.columns else x_axis

        for idx, sensor in enumerate(active_sensors):
            if sensor in scores.columns:
                color = palette[idx % len(palette)]
                fig_scores.add_trace(go.Scatter(
                    x=scores_x,
                    y=scores[sensor],
                    mode="lines",
                    name=f"{sensor} Score",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{sensor} Score</b>: %{{y:.3f}}<extra></extra>",
                ))

        # Sustained alarm threshold reference line
        fig_scores.add_hline(
            y=0.6,
            line_dash="dash",
            line_color=ACCENT_RED,
            annotation_text="Alarm Threshold (0.60)",
            annotation_position="top right",
            annotation_font=dict(color=ACCENT_RED, size=11),
        )

        fig_scores.update_layout(
            title="Calibrated Multivariate Anomaly Scores [0.0 - 1.0]",
            xaxis_title="Simulation Time Step",
            yaxis_title="Anomaly Score",
            yaxis_range=[-0.05, 1.05],
            template="plotly_dark",
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_PLOT_BG,
            hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scores, use_container_width=True)


def render_network_graph(
    st: Any,
    network_graph: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
    onsets: Optional[Dict[str, Optional[int]]],
) -> None:
    """Visualize the NetworkX sensor network topology using Plotly.

    Validates graph input and warns if invalid rather than constructing
    a default graph silently. Nodes are highlighted by diagnosis status.

    Args:
        st: Streamlit module or container.
        network_graph: SensorNetworkGraph instance or NetworkX Graph.
        rankings: Ranked suspect tuples from FaultLocalizer.
        onsets: Detection onset timestep dictionary per sensor.
    """
    if hasattr(network_graph, "graph") and isinstance(network_graph.graph, nx.Graph):
        G = network_graph.graph
    elif isinstance(network_graph, nx.Graph):
        G = network_graph
    else:
        st.warning("Sensor network graph is missing or invalid.")
        return

    if len(G.nodes) == 0:
        st.warning("Sensor network graph contains no nodes.")
        return

    # Fixed deterministic 2D spring layout
    pos = nx.spring_layout(G, seed=42)

    # 1. Edge Coordinates
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2.5, color="#3a506b"),
        hoverinfo="none",
        name="Physical Coupling",
    )

    # 2. Node Attributes & Highlighting
    top_suspect = rankings[0][0] if rankings and len(rankings) > 0 else None
    ranking_lookup = {r[0]: (idx + 1, r[1], r[2]) for idx, r in enumerate(rankings)} if rankings else {}
    onsets_lookup = onsets or {}

    node_x: List[float] = []
    node_y: List[float] = []
    node_labels: List[str] = []
    node_hover: List[str] = []
    node_colors: List[str] = []
    node_sizes: List[int] = []
    node_symbols: List[str] = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        onset_val = onsets_lookup.get(node)
        rank_data = ranking_lookup.get(node)

        neighbors = list(G.neighbors(node))
        neighbors_str = ", ".join(neighbors) if neighbors else "None"

        if node == top_suspect:
            # Highlight Top-1 Suspect
            node_colors.append(ACCENT_RED)
            node_sizes.append(38)
            node_symbols.append("diamond")
            node_labels.append(f"<b>{node} (TOP-1)</b>")
            status = "PRIMARY ROOT CAUSE"
        elif onset_val is not None:
            # Alarmed coupled neighbor
            node_colors.append(ACCENT_AMBER)
            node_sizes.append(28)
            node_symbols.append("circle")
            node_labels.append(f"{node} (t={onset_val})")
            status = "Coupled Alarm"
        else:
            # Nominal sensor
            node_colors.append(ACCENT_CYAN)
            node_sizes.append(22)
            node_symbols.append("circle")
            node_labels.append(f"{node}")
            status = "Nominal Baseline"

        rank_str = f"#{rank_data[0]}" if rank_data else "N/A"
        score_str = f"{rank_data[1]:.4f}" if rank_data else "0.0000"
        onset_str = f"t={onset_val}" if onset_val is not None else "None"

        hover_info = (
            f"<b>Sensor {node}</b><br>"
            f"Classification: {status}<br>"
            f"Diagnosis Rank: {rank_str}<br>"
            f"Root-Cause Score: {score_str}<br>"
            f"Detection Onset: {onset_str}<br>"
            f"Topological Neighbors: {neighbors_str}"
        )
        node_hover.append(hover_info)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=2, color="#ffffff"),
        ),
        name="Sensors",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Mesh Topology & Physical Coupling Graph",
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_PLOT_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_localization_table(
    st: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Display root-cause suspect rankings, composite scores, and diagnostic explanations.

    Args:
        st: Streamlit module or container.
        rankings: Ranked suspect tuples from FaultLocalizer.
    """
    if not rankings:
        st.info("No localization diagnosis available.")
        return

    top_sensor, top_score, top_bd = rankings[0]
    st.success(f"**Primary Root Cause Identified:** Sensor **{top_sensor}** (Composite Score: **{top_score:.4f}**)")

    table_data = []
    for rank, (sensor, composite, breakdown) in enumerate(rankings, start=1):
        onset_val = breakdown.get("onset")
        onset_str = f"t={onset_val}" if onset_val is not None else "None"

        table_data.append({
            "Rank": f"#{rank}",
            "Sensor": sensor,
            "Root-Cause Score": f"{composite:.4f}",
            "Own Anomaly (35%)": f"{breakdown.get('own_anomaly', 0.0):.3f}",
            "Earliness (30%)": f"{breakdown.get('detection_earliness', 0.0):.3f}",
            "Neighbor Support (20%)": f"{breakdown.get('neighbour_support', 0.0):.3f}",
            "Coverage (15%)": f"{breakdown.get('coverage', 0.0):.3f}",
            "Detection Onset": onset_str,
            "Diagnostic Explanation": breakdown.get("explanation", ""),
        })

    df_rankings = pd.DataFrame(table_data)
    st.dataframe(df_rankings, use_container_width=True, hide_index=True)


def render_benchmark_summary(
    st: Any,
    benchmark: Optional[Dict[str, Any]],
) -> None:
    """Display evaluated benchmark metrics from load_benchmark().

    Never hardcodes accuracy. Dynamically reports measured performance
    across all evaluated benchmark scenarios.

    Args:
        st: Streamlit module or container.
        benchmark: Benchmark dictionary parsed from data/evaluation_results.csv.
    """
    if not benchmark:
        st.warning("Benchmark results not found at `data/evaluation_results.csv`. Run `python evaluate.py` to generate the benchmark evaluation.")
        return

    # Top KPI Metrics
    total = benchmark.get("total_scenarios", 0)
    top1_acc = benchmark.get("top1_accuracy", 0.0)
    top2_acc = benchmark.get("top2_accuracy", 0.0)
    det_rate = benchmark.get("detection_rate", 0.0)
    top1_correct = benchmark.get("top1_correct", 0)
    top2_correct = benchmark.get("top2_correct", 0)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Evaluated Scenarios", f"{total}", delta="5 Sensors x 3 Faults x 5 Seeds")
    b2.metric("Top-1 Accuracy", f"{top1_acc:.2f}%", delta=f"{top1_correct} / {total} correct")
    b3.metric("Top-2 Accuracy", f"{top2_acc:.2f}%", delta=f"{top2_correct} / {total} correct")
    b4.metric("Injected Detection Rate", f"{det_rate:.2f}%", delta="Alarm Confirmation")

    # Fault-Type Breakdown Table
    fault_breakdown = benchmark.get("fault_type_accuracy", {})
    if fault_breakdown:
        st.markdown("#### Performance by Fault Type")
        ft_rows = []
        for ft, data in fault_breakdown.items():
            ft_rows.append({
                "Fault Type": ft.capitalize(),
                "Scenarios": data.get("total", 0),
                "Top-1 Correct": data.get("top1_correct", 0),
                "Top-1 Accuracy (%)": f"{data.get('top1_accuracy', 0.0):.2f}%",
                "Top-2 Correct": data.get("top2_correct", 0),
                "Top-2 Accuracy (%)": f"{data.get('top2_accuracy', 0.0):.2f}%",
            })
        st.dataframe(pd.DataFrame(ft_rows), use_container_width=True, hide_index=True)

    # Sensor Breakdown Table
    sensor_breakdown = benchmark.get("sensor_accuracy", {})
    if sensor_breakdown:
        st.markdown("#### Performance by Injected Sensor")
        s_rows = []
        for s, data in sensor_breakdown.items():
            s_rows.append({
                "Injected Sensor": s,
                "Scenarios": data.get("total", 0),
                "Top-1 Correct": data.get("top1_correct", 0),
                "Top-1 Accuracy (%)": f"{data.get('top1_accuracy', 0.0):.2f}%",
                "Top-2 Correct": data.get("top2_correct", 0),
                "Top-2 Accuracy (%)": f"{data.get('top2_accuracy', 0.0):.2f}%",
            })
        st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    # Full Dataset Expander with dynamic scenario count
    if "df" in benchmark and isinstance(benchmark["df"], pd.DataFrame):
        with st.expander(f"Explore Complete {total}-Scenario Evaluation Dataset"):
            st.dataframe(benchmark["df"], use_container_width=True, hide_index=True)


def render_download_controls(
    st: Any,
    readings: Optional[pd.DataFrame],
    scores: Optional[pd.DataFrame],
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Provide download buttons for readings, anomaly scores, and diagnosis reports.

    Ground-truth fault parameters attached in df.attrs are not exported to
    maintain clean separation between operational data and hidden labels.

    Args:
        st: Streamlit module or container.
        readings: Sensor readings DataFrame.
        scores: Anomaly scores DataFrame.
        rankings: Ranked suspect tuples from FaultLocalizer.
    """
    col1, col2, col3 = st.columns(3)

    # 1. Readings CSV
    if readings is not None and not readings.empty:
        clean_readings = readings.copy()
        readings_csv = clean_readings.to_csv(index=False).encode("utf-8")
        col1.download_button(
            label="Download Sensor Readings (CSV)",
            data=readings_csv,
            file_name="sensor_readings.csv",
            mime="text/csv",
        )
    else:
        col1.button("Download Sensor Readings", disabled=True)

    # 2. Scores CSV
    if scores is not None and not scores.empty:
        scores_csv = scores.to_csv(index=False).encode("utf-8")
        col2.download_button(
            label="Download Anomaly Scores (CSV)",
            data=scores_csv,
            file_name="anomaly_scores.csv",
            mime="text/csv",
        )
    else:
        col2.button("Download Anomaly Scores", disabled=True)

    # 3. Diagnosis Rankings CSV
    if rankings and len(rankings) > 0:
        report_rows = []
        for rank, (sensor, composite, breakdown) in enumerate(rankings, start=1):
            report_rows.append({
                "rank": rank,
                "sensor": sensor,
                "composite_score": round(composite, 4),
                "own_anomaly": round(breakdown.get("own_anomaly", 0.0), 4),
                "detection_earliness": round(breakdown.get("detection_earliness", 0.0), 4),
                "neighbour_support": round(breakdown.get("neighbour_support", 0.0), 4),
                "coverage": round(breakdown.get("coverage", 0.0), 4),
                "onset_timestep": breakdown.get("onset"),
                "diagnostic_explanation": breakdown.get("explanation", ""),
            })
        report_csv = pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8")
        col3.download_button(
            label="Download Diagnosis Report (CSV)",
            data=report_csv,
            file_name="faultmesh_diagnosis_report.csv",
            mime="text/csv",
        )
    else:
        col3.button("Download Diagnosis Report", disabled=True)
