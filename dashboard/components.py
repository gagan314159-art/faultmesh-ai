"""Reusable Streamlit and Plotly dashboard components for FaultMesh AI.

Responsibility:
    Provides modular, independent rendering components for KPI cards, sensor
    time-series, anomaly scores, network topology, localization ranking tables,
    benchmark performance summaries, and report download controls.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import io
import os
import json
import sqlite3
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
ACCENT_PURPLE = "#ab47bc"


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
    diagnosis_status: Optional[str] = None,
    primary_suspect: Optional[str] = None,
) -> None:
    """Display root-cause suspect rankings, composite scores, and diagnostic explanations.

    Args:
        st: Streamlit module or container.
        rankings: Ranked suspect tuples from FaultLocalizer.
        diagnosis_status: Optional diagnostic state ('NORMAL', 'UNCERTAIN', 'CONFIRMED').
        primary_suspect: Optional confirmed initiating root-cause sensor ID.
    """
    if not rankings:
        st.info("No localization diagnosis available.")
        return

    top_sensor, top_score, top_bd = rankings[0]
    has_alarms = any(bd.get("onset") is not None for _, _, bd in rankings)
    max_own = max(bd.get("own_anomaly", 0.0) for _, _, bd in rankings)

    if diagnosis_status == "NORMAL" or (diagnosis_status is None and not has_alarms and max_own < 0.50 and top_score < 0.40):
        st.info("✅ **Nominal Mesh Condition:** All sensors operating within calibrated normal limits. No active fault detected.")
    elif diagnosis_status == "UNCERTAIN" or (diagnosis_status is None and not has_alarms and max_own < 0.60):
        st.warning("⚠️ **Root Cause Uncertain:** Mildly elevated readings detected, but temporal or propagation evidence is insufficient to isolate a definitive root cause.")
    else:
        suspect = primary_suspect or top_sensor
        st.success(f"🎯 **Primary Root Cause Identified:** Sensor **{suspect}** (Composite Score: **{top_score:.4f}**)")

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


def render_explainable_investigation(
    st: Any,
    investigation_report: Any,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]],
) -> None:
    """Render Explainable Fault Investigation Engine diagnostic UI.

    Displays:
    1. Suspected root-cause sensor.
    2. Investigation confidence category.
    3. Anomaly evidence.
    4. Detection timing comparison.
    5. Supporting neighboring sensors.
    6. Explanation of the localization ranking.
    7. Final investigation summary.
    """
    st.subheader("Explainable Fault Investigation Engine")

    if not investigation_report:
        st.info(
            "**Explainable Fault Investigation Engine Active:** "
            "Awaiting incoming telemetry stream and baseline evaluation to generate causal graph evidence."
        )
        return

    suspect = getattr(investigation_report, "suspected_sensor", None)
    confidence = getattr(investigation_report, "confidence_category", "Insufficient")
    loc_score = getattr(investigation_report, "localization_score", 0.0)
    anom_score = getattr(investigation_report, "anomaly_score", 0.0)
    onset = getattr(investigation_report, "detection_onset", None)

    # 1. Suspected Root-Cause Sensor & 2. Confidence Category
    col1, col2, col3, col4 = st.columns(4)
    if suspect:
        suspect_display = f"Sensor {suspect}"
        delta_rank = "Rank #1 (Suspect)"
    elif confidence == "Limited":
        suspect_display = "Root Cause Uncertain"
        delta_rank = "Ambiguous Cascade"
    else:
        suspect_display = "None (Nominal)"
        delta_rank = "Within Baseline"

    col1.metric(label="Suspected Root Cause", value=suspect_display, delta=delta_rank)
    col2.metric(label="Investigation Confidence", value=confidence, delta="Evidence Strength")
    col3.metric(label="Localization Score", value=f"{loc_score:.4f}", delta="Composite Fusion")
    col4.metric(label="Detection Onset", value=f"t = {onset}" if onset is not None else "None", delta=f"Peak Anomaly: {anom_score:.3f}")

    st.divider()

    # 3. Anomaly Evidence
    st.markdown("### 1. Observed Anomaly Evidence")
    evidence_list = getattr(investigation_report, "evidence", [])
    if evidence_list:
        for ev in evidence_list:
            st.markdown(f"- {ev}")
    else:
        st.caption("No specific anomaly evidence generated.")

    # 4. Detection Timing Comparison
    st.markdown("### 2. Detection Timing Comparison Across Sensors")
    timing_data = getattr(investigation_report, "timing_comparisons", [])
    if timing_data:
        timing_rows = []
        for t in timing_data:
            timing_rows.append({
                "Sensor ID": t.get("sensor"),
                "Onset Timestep": f"t = {t.get('onset')}" if t.get("onset") is not None else "None",
                "Relative Delay": t.get("delay_relative_to_suspect"),
                "Diagnostic Timing Status": t.get("timing_status"),
                "Peak Anomaly Score": f"{t.get('anomaly_score', 0.0):.3f}",
            })
        st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No timing comparison data available.")

    # 5. Supporting Neighboring Sensors
    st.markdown("### 3. Supporting Neighboring Sensor Analysis")
    neighbor_data = getattr(investigation_report, "neighbor_evidence", [])
    supporting_list = getattr(investigation_report, "supporting_sensors", [])

    if neighbor_data:
        st.markdown(f"**Supporting Neighbor Nodes:** {', '.join(supporting_list) if supporting_list else 'None'}")
        n_rows = []
        for n in neighbor_data:
            n_rows.append({
                "Connected Neighbor": n.get("neighbor"),
                "Onset Timestep": f"t = {n.get('onset')}" if n.get("onset") is not None else "None",
                "Anomaly Score": f"{n.get('anomaly_score', 0.0):.3f}",
                "Topological Causal Relation": n.get("relationship"),
                "Supports Root Cause?": "YES (Delayed cascade)" if n.get("supports_suspect") else "NO",
            })
        st.dataframe(pd.DataFrame(n_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No neighboring sensor evidence available.")

    # 6. Explanation of Localization Ranking
    st.markdown("### 4. Explanation of Localization Ranking Criteria")
    breakdown = getattr(investigation_report, "ranking_breakdown", {})
    if breakdown:
        c_own, c_early, c_supp, c_cov = st.columns(4)
        c_own.metric("Own Anomaly (35%)", f"{breakdown.get('own_anomaly', 0.0):.3f}")
        c_early.metric("Earliness (30%)", f"{breakdown.get('detection_earliness', 0.0):.3f}")
        c_supp.metric("Neighbor Support (20%)", f"{breakdown.get('neighbour_support', 0.0):.3f}")
        c_cov.metric("Graph Coverage (15%)", f"{breakdown.get('coverage', 0.0):.3f}")

        st.caption(
            f"Sensor {suspect} was ranked #1 with composite score {loc_score:.4f} by combining "
            f"individual anomaly magnitude ({breakdown.get('own_anomaly', 0.0):.3f}), onset precedence ({breakdown.get('detection_earliness', 0.0):.3f}), "
            f"coupled peer abnormalities ({breakdown.get('neighbour_support', 0.0):.3f}), and mesh coverage ({breakdown.get('coverage', 0.0):.3f})."
        )

    # 7. Final Investigation Summary
    st.markdown("### 5. Final Fault Investigation Summary Report")
    summary_text = getattr(investigation_report, "investigation_summary", "")
    if summary_text:
        st.code(summary_text, language="text")

    # Limitations Expander
    limitations = getattr(investigation_report, "limitations", [])
    if limitations:
        with st.expander("Diagnostic Limitations & Assumptions"):
            for lim in limitations:
                st.markdown(f"- {lim}")


def render_sensor_health_center(
    st: Any,
    health_reports: Optional[Dict[str, Any]],
    network_summary: Optional[Any] = None,
) -> None:
    """Render Sensor Health Center dashboard section.

    Displays:
    1. Overall network health score.
    2. Individual sensor health cards.
    3. Health status indicators.
    4. Sensor health comparison chart (Plotly).
    5. Anomaly frequency visualization (Plotly).
    6. Sensor inspection priority table.
    7. Health explanations.
    """
    st.subheader("Sensor Health Center")

    if not health_reports:
        st.info("Execute a simulation to view sensor health diagnostics.")
        return

    # Calculate summary if not passed
    if network_summary is None:
        from faultmesh.health import SensorHealthMonitor
        monitor = SensorHealthMonitor()
        summary = monitor.summarize_network_health(health_reports)
    else:
        summary = network_summary

    # 1. Overall Network Health Score & 3. Health Status Indicators
    st.markdown("### Network Health Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Network Health Index", f"{summary.overall_network_health_score:.1f} / 100")
    kpi2.metric("Healthy Nodes", f"{summary.healthy_count} / 5")
    kpi3.metric("Warning Nodes", f"{summary.warning_count} / 5")
    kpi4.metric("Critical Nodes", f"{summary.critical_count} / 5")
    kpi5.metric("Insufficient Data", f"{summary.insufficient_data_count} Nodes")
    vulnerable_val = f"Sensor {summary.most_vulnerable_sensor}" if summary.most_vulnerable_sensor else "None"
    kpi6.metric("Most Vulnerable", vulnerable_val)

    # Engineering Disclaimer
    st.caption("ℹ️ Health scores evaluate simulated anomaly frequency, baseline drift, variance stability, and onset duration.")

    st.divider()

    # 2. Individual Sensor Health Cards
    st.markdown("### Individual Sensor Health Cards")
    card_cols = st.columns(5)
    sorted_sensors = sorted(health_reports.keys())

    for idx, s_id in enumerate(sorted_sensors):
        rep = health_reports[s_id]
        with card_cols[idx]:
            # Visual status indicator color
            status = getattr(rep, "health_status", getattr(rep, "status", "Healthy"))
            score = getattr(rep, "health_score", 0.0)
            priority = getattr(rep, "recommended_monitoring_priority", "Normal")
            curr_val = getattr(rep, "current_value", None)
            unit = getattr(rep, "unit", "")

            status_delta = f"Status: {status}"
            delta_color = "normal" if status == "Healthy" else "inverse"

            st.metric(
                label=f"{s_id} ({rep.sensor_name})",
                value=f"{score:.1f} / 100",
                delta=status_delta,
                delta_color=delta_color,
            )
            st.caption(f"**Priority:** {priority}")
            if curr_val is not None:
                st.caption(f"**Current:** {curr_val} {unit}")

    st.divider()

    # 4. Sensor Health Comparison Chart & 5. Anomaly Frequency Visualization
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Sensor Health Score Comparison")
        fig_health = go.Figure()

        sensors = list(sorted_sensors)
        scores = [getattr(health_reports[s], "health_score", 0.0) for s in sensors]
        statuses = [getattr(health_reports[s], "health_status", getattr(health_reports[s], "status", "Healthy")) for s in sensors]

        bar_colors = []
        for stat in statuses:
            if stat == "Healthy":
                bar_colors.append(ACCENT_GREEN)
            elif stat == "Warning":
                bar_colors.append(ACCENT_AMBER)
            elif stat == "Critical":
                bar_colors.append(ACCENT_RED)
            else:
                bar_colors.append("#94a3b8")

        fig_health.add_trace(go.Bar(
            x=sensors,
            y=scores,
            marker_color=bar_colors,
            text=[f"{sc:.1f}" for sc in scores],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Health Score: %{y:.1f}<extra></extra>",
        ))

        # Add threshold reference lines
        fig_health.add_hline(y=85.0, line_dash="dash", line_color=ACCENT_GREEN, annotation_text="Healthy (85)", annotation_position="top right")
        fig_health.add_hline(y=65.0, line_dash="dash", line_color=ACCENT_AMBER, annotation_text="Warning (65)", annotation_position="bottom right")

        fig_health.update_layout(
            yaxis=dict(range=[0, 105], title="Health Score (0-100)"),
            xaxis=dict(title="Monitored Sensor Channel"),
            template="plotly_dark",
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_PLOT_BG,
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
        )
        st.plotly_chart(fig_health, use_container_width=True)

    with chart_col2:
        st.markdown("### Anomaly Frequency (% Abnormal Window)")
        fig_freq = go.Figure()

        freqs = [getattr(health_reports[s], "anomaly_frequency", 0.0) * 100.0 for s in sensors]
        fig_freq.add_trace(go.Bar(
            x=sensors,
            y=freqs,
            marker_color=ACCENT_CYAN,
            text=[f"{f:.1f}%" for f in freqs],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Anomaly Frequency: %{y:.1f}%<extra></extra>",
        ))

        fig_freq.update_layout(
            yaxis=dict(title="Abnormal Samples (%)"),
            xaxis=dict(title="Monitored Sensor Channel"),
            template="plotly_dark",
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_PLOT_BG,
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # 6. Sensor Inspection Priority Table
    st.markdown("### Sensor Inspection Priority Rankings")
    # Sort sensors by lowest health score first (highest inspection priority)
    sorted_by_priority = sorted(
        health_reports.values(),
        key=lambda r: (r.health_score if getattr(r, "health_status", getattr(r, "status", "")) != "Insufficient Data" else 999.0)
    )

    p_rows = []
    for rank_idx, rep in enumerate(sorted_by_priority, start=1):
        onset_str = f"t = {rep.detection_onset}" if getattr(rep, "detection_onset", None) is not None else "None"
        p_rows.append({
            "Priority Rank": f"#{rank_idx}",
            "Sensor ID": getattr(rep, "sensor_id", getattr(rep, "sensor", "")),
            "Parameter": getattr(rep, "sensor_name", ""),
            "Health Score": f"{rep.health_score:.1f} / 100",
            "Health Status": getattr(rep, "health_status", getattr(rep, "status", "")),
            "Inspection Action": getattr(rep, "recommended_monitoring_priority", "Normal"),
            "Peak Anomaly Score": f"{getattr(rep, 'maximum_anomaly_score', 0.0):.3f}",
            "Alarm Onset": onset_str,
            "Recent Trend": getattr(rep, "recent_trend", "Stable"),
        })
    st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

    # 7. Health Explanations
    st.markdown("### Diagnostic Health Explanations")
    for s_id in sorted_sensors:
        rep = health_reports[s_id]
        stat = getattr(rep, "health_status", getattr(rep, "status", "Healthy"))
        with st.expander(f"Sensor {s_id} ({rep.sensor_name}) -- {stat} ({rep.health_score:.1f}/100)"):
            st.write(f"**Explanation:** {rep.health_explanation}")
            factors = getattr(rep, "degradation_factors", [])
            if factors:
                st.markdown("**Degradation Drivers:**")
                for factor in factors:
                    st.markdown(f"- {factor}")


# Maintain alias for backward compatibility
render_sensor_health = render_sensor_health_center


def render_predictive_maintenance_center(
    st: Any,
    advisories: Optional[Dict[str, Any]],
    maintenance_summary: Optional[Any] = None,
    health_reports: Optional[Dict[str, Any]] = None,
    primary_suspect: Optional[str] = None,
) -> None:
    """Render Predictive Maintenance Center dashboard section.

    Displays:
    1. Overall maintenance attention level.
    2. Maintenance priority summary cards.
    3. Sensor-wise maintenance priority table.
    4. Recommended actions.
    5. Evidence behind each recommendation.
    6. Maintenance priority comparison chart (Plotly).
    7. Sensor health versus maintenance priority visualization (Plotly).
    8. Sensors requiring additional data.
    """
    st.subheader("Predictive Maintenance Center")

    if not advisories:
        st.info("Execute a simulation to generate evidence-based predictive maintenance recommendations.")
        return

    # Derive maintenance summary if not provided
    if maintenance_summary is None:
        from faultmesh.maintenance import MaintenanceAdvisor
        advisor = MaintenanceAdvisor()
        summary = advisor.summarize_network_maintenance(advisories)
    else:
        summary = maintenance_summary

    # Engineering Disclaimer Notice
    first_adv = next(iter(advisories.values()), None)
    if first_adv and hasattr(first_adv, "safety_disclaimer"):
        st.caption(f"ℹ️ **Engineering Disclaimer:** {first_adv.safety_disclaimer}")

    st.divider()

    # 1. Overall Maintenance Attention Level & 2. Maintenance Priority Summary Cards
    st.markdown("### Maintenance Planning Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)

    attention = getattr(summary, "overall_maintenance_attention_level", "Nominal")
    delta_color = "inverse" if attention in ("Critical", "High") else "normal"
    kpi1.metric("Fleet Attention", attention, delta="Attention Level", delta_color=delta_color)
    kpi2.metric("Critical Priority", f"{getattr(summary, 'critical_cases', 0)} Nodes")
    kpi3.metric("High Priority", f"{getattr(summary, 'high_priority_cases', 0)} Nodes")
    kpi4.metric("Medium Priority", f"{getattr(summary, 'medium_priority_cases', 0)} Nodes")
    kpi5.metric("Low Priority", f"{getattr(summary, 'low_priority_cases', 0)} Nodes")
    kpi6.metric("Data Required", f"{getattr(summary, 'insufficient_data_cases', 0)} Nodes")
    highest_s = getattr(summary, "highest_priority_sensor", "None")
    kpi7.metric("Top Priority Node", f"Sensor {highest_s}" if highest_s else "None")

    st.divider()

    # 6. Maintenance Priority Comparison Chart & 7. Sensor Health vs Maintenance Priority Visualization
    chart_col1, chart_col2 = st.columns(2)

    sorted_sensor_keys = sorted(advisories.keys())

    with chart_col1:
        st.markdown("### Maintenance Urgency Index by Sensor")
        fig_urgency = go.Figure()

        sensors = list(sorted_sensor_keys)
        urgencies = [getattr(advisories[s], "urgency_score", 0.0) for s in sensors]
        priorities = [getattr(advisories[s], "priority", "Low") for s in sensors]

        p_colors = []
        for p in priorities:
            if p == "Critical":
                p_colors.append(ACCENT_RED)
            elif p == "High":
                p_colors.append(ACCENT_AMBER)
            elif p == "Medium":
                p_colors.append(ACCENT_BLUE)
            elif p == "Low":
                p_colors.append(ACCENT_GREEN)
            else:
                p_colors.append("#94a3b8")

        fig_urgency.add_trace(go.Bar(
            x=sensors,
            y=urgencies,
            marker_color=p_colors,
            text=[f"{u:.1f}" for u in urgencies],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Priority: %{customdata}<br>Urgency Score: %{y:.1f}<extra></extra>",
            customdata=priorities,
        ))

        fig_urgency.add_hline(y=75.0, line_dash="dash", line_color=ACCENT_RED, annotation_text="Critical (75)", annotation_position="top right")
        fig_urgency.add_hline(y=50.0, line_dash="dash", line_color=ACCENT_AMBER, annotation_text="High (50)", annotation_position="bottom right")

        fig_urgency.update_layout(
            yaxis=dict(range=[0, 105], title="Urgency Score (0-100)"),
            xaxis=dict(title="Monitored Sensor Channel"),
            template="plotly_dark",
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_PLOT_BG,
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
        )
        st.plotly_chart(fig_urgency, use_container_width=True)

    with chart_col2:
        st.markdown("### Sensor Health vs. Maintenance Urgency")
        fig_dual = go.Figure()

        # Extract health scores if health reports available, else fallback
        health_scores = []
        for s in sensors:
            if health_reports and s in health_reports:
                h_val = getattr(health_reports[s], "health_score", 100.0)
            elif hasattr(advisories[s], "health_metrics") and "health_score" in advisories[s].health_metrics:
                h_val = advisories[s].health_metrics["health_score"]
            else:
                h_val = 100.0 - urgencies[sensors.index(s)]
            health_scores.append(h_val)

        fig_dual.add_trace(go.Bar(
            name="Health Score",
            x=sensors,
            y=health_scores,
            marker_color=ACCENT_CYAN,
            text=[f"{h:.1f}" for h in health_scores],
            textposition="auto",
        ))

        fig_dual.add_trace(go.Bar(
            name="Maintenance Urgency",
            x=sensors,
            y=urgencies,
            marker_color=ACCENT_PURPLE,
            text=[f"{u:.1f}" for u in urgencies],
            textposition="auto",
        ))

        fig_dual.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 105], title="Score (0-100)"),
            xaxis=dict(title="Sensor Channel"),
            template="plotly_dark",
            paper_bgcolor=THEME_BG,
            plot_bgcolor=THEME_PLOT_BG,
            margin=dict(l=40, r=20, t=30, b=40),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    st.divider()

    # 3. Sensor-Wise Maintenance Priority Table
    st.markdown("### Sensor Maintenance Priority Schedule")
    table_rows = []

    # Order table by urgency descending
    sorted_advisories = sorted(advisories.values(), key=lambda a: getattr(a, "urgency_score", 0.0), reverse=True)

    for rank_idx, adv in enumerate(sorted_advisories, start=1):
        f_mode = getattr(adv, "primary_fault_mode", None)
        mode_str = f_mode.upper() if f_mode else "Nominal"
        table_rows.append({
            "Priority Rank": f"#{rank_idx}",
            "Sensor ID": getattr(adv, "sensor_id", getattr(adv, "sensor", "")),
            "Parameter": getattr(adv, "sensor_name", ""),
            "Maintenance Priority": getattr(adv, "priority", "Low"),
            "Recommended Action": getattr(adv, "recommended_action", "Routine maintenance review"),
            "Urgency Score": f"{getattr(adv, 'urgency_score', 0.0):.1f} / 100",
            "Follow-up Window": getattr(adv, "suggested_follow_up", "Routine (30 days)"),
            "Fault Signature": mode_str,
            "Reason": getattr(adv, "reason", ""),
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 8. Sensors Requiring Additional Data Callout
    insufficient_nodes = getattr(summary, "sensors_requiring_more_data", [])
    if insufficient_nodes:
        st.warning(
            f"⚠️ **Telemetry Alert -- Insufficient Data on {len(insufficient_nodes)} Sensor(s):** "
            f"`{', '.join(insufficient_nodes)}`. Recommended Action: Initiate physical DAQ wiring and loop continuity checks."
        )

    # 4. Recommended Actions & 5. Evidence Behind Each Recommendation
    st.markdown("### Prescribed Maintenance Protocols & Evidence Details")

    for adv in sorted_advisories:
        s_id = getattr(adv, "sensor_id", getattr(adv, "sensor", ""))
        p_label = getattr(adv, "priority", "Low")
        score_val = getattr(adv, "urgency_score", 0.0)
        action_val = getattr(adv, "recommended_action", "Routine Maintenance")

        expander_title = f"Sensor {s_id} ({adv.sensor_name}) -- [{p_label} Priority] {action_val} (Urgency: {score_val:.1f}/100)"
        default_open = (p_label in ("Critical", "High"))

        with st.expander(expander_title, expanded=default_open):
            st.markdown(f"**Diagnostic Reason:** {adv.reason}")

            # Evidence Breakdown
            st.markdown("#### 🔬 Supporting Evidence")
            ev_list = getattr(adv, "supporting_evidence", getattr(adv, "evidence", []))
            if ev_list:
                for ev in ev_list:
                    st.markdown(f"- {ev}")
            else:
                st.caption("No specific anomaly evidence recorded.")

            # Prescribed Maintenance Actions
            action_items = getattr(adv, "actions", [])
            if action_items:
                st.markdown("#### 🛠️ Recommended Action Items")
                for act in action_items:
                    with st.container():
                        st.markdown(f"**[{act.action_id}] {act.title}** (`{act.component}`)")
                        st.write(act.description)
                        st.write(f"🔧 **Required Equipment:** {', '.join(act.required_tools)}")
                        st.caption(f"⚠️ **Safety Notes:** {act.safety_notes}")

            # Verification Checklist
            chk_list = getattr(adv, "inspection_checklist", [])
            if chk_list:
                st.markdown("#### 📋 Technician Verification Checklist")
                for idx, chk in enumerate(chk_list):
                    st.checkbox(chk, key=f"maint_chk_{s_id}_{idx}")


# Maintain aliases for backward compatibility
render_predictive_maintenance = render_predictive_maintenance_center


def render_executive_reports_tab(
    st: Any,
    summary: Optional[Dict[str, Any]],
    readings: Optional[pd.DataFrame] = None,
    scores: Optional[pd.DataFrame] = None,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
    benchmark: Optional[Dict[str, Any]] = None,
    db_path: str = "data/faultmesh.db",
) -> None:
    """Render Executive Reporting & PDF Export Center dashboard tab.

    Displays:
    1. Report overview and executive diagnosis synopsis.
    2. Latest investigation summary KPIs.
    3. Sensor-wise diagnostic findings and health metrics table.
    4. Maintenance priority summary and technician procedures.
    5. Publication-grade PDF report generation and download.
    6. CSV diagnostic summary and raw telemetry export.
    7. Clear messages and placeholders when no simulation has been executed.
    8. Historical investigation reports retrieved from SQLite persistence.
    """
    st.subheader("Executive Reporting & PDF Export Center")
    from faultmesh.reporting import SENSOR_SUBSYSTEMS, generate_pdf_report, export_summary_csv

    # 1. Check if simulation has been executed
    if not summary or readings is None or readings.empty:
        st.info(
            "ℹ️ **Ready to Generate Executive Report:** Configure simulation parameters in the sidebar "
            "and click **Run Simulation** to compile an auditable, publication-grade investigation report."
        )

        # Still render Historical Reports from SQLite even before simulation
        _render_sqlite_report_history(st, db_path)
        return

    # Extract metadata
    r_id = summary.get("report_id", summary.get("metadata", {}).get("report_id", "RPT-PENDING"))
    gen_time = summary.get("generated_at", summary.get("metadata", {}).get("generated_at", ""))
    sim_cfg = summary.get("simulation_config", {})
    injected_summary = sim_cfg.get("injected_fault_summary", "Clean Nominal Baseline")

    loc = summary.get("localization", {})
    primary = loc.get("primary_suspect") or "None (Nominal)"
    score = loc.get("composite_score", 0.0)
    conf = loc.get("confidence_level", "Moderate")

    health_info = summary.get("sensor_health", {})
    net_score = health_info.get("network_health_score", 100.0)

    maint_info = summary.get("maintenance_summary", {})
    attention = maint_info.get("overall_attention_level", "Nominal")

    # 2. Latest Investigation Summary KPI Banner
    st.markdown("### Executive Investigation Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Report ID", r_id[:16])
    suspect_display = f"Sensor {primary}" if primary != "None (Nominal)" else primary
    kpi2.metric("Localized Suspect", suspect_display, delta="Rank #1")
    kpi3.metric("Localization Score", f"{score:.4f}", delta="Composite Fusion")
    kpi4.metric("Network Health", f"{net_score:.1f} / 100", delta=f"{conf} Conf.")
    delta_attn = "inverse" if attention in ("Critical", "High") else "normal"
    kpi5.metric("Fleet Attention", attention, delta="Maintenance Level", delta_color=delta_attn)

    # Narrative Synopsis Callout
    exec_summary_text = summary.get("executive_summary", "")
    if exec_summary_text:
        st.info(f"**Diagnostic Synopsis:** {exec_summary_text}")

    st.divider()

    # 3. Sensor-Wise Diagnostic Findings Table
    st.markdown("### Sensor-Wise Diagnostic Findings & Health Overview")
    health_reports = summary.get("sensor_health", {}).get("reports", summary.get("health_reports", {}))
    stats = summary.get("anomaly_detection", {}).get("telemetry_stats", summary.get("telemetry_stats", {}))
    advisories = summary.get("predictive_maintenance", {}).get("advisories", summary.get("advisories", {}))
    rankings_list = loc.get("rankings", [])
    rankings_dict = {item[0]: (i + 1, item[1]) for i, item in enumerate(rankings_list)}

    table_rows = []
    sensors = ["S1", "S2", "S3", "S4", "S5"]
    for s_id in sensors:
        rep = health_reports.get(s_id)
        st_data = stats.get(s_id, {})
        adv = advisories.get(s_id)
        rank_info = rankings_dict.get(s_id, ("Nominal", 0.0))
        meta = SENSOR_SUBSYSTEMS.get(s_id, {})

        table_rows.append({
            "Sensor ID": s_id,
            "Subsystem": meta.get("name", s_id),
            "Suspect Rank": f"#{rank_info[0]}" if isinstance(rank_info[0], int) else rank_info[0],
            "Alarm Onset": f"t = {st_data.get('alarm_onset')}" if st_data.get("alarm_onset") is not None else "None",
            "Peak Anomaly": f"{st_data.get('peak_anomaly_score', 0.0):.3f}",
            "Health Score": f"{getattr(rep, 'health_score', 100.0):.1f} / 100" if rep else "N/A",
            "Health Status": getattr(rep, "health_status", getattr(rep, "status", "Healthy")) if rep else "Healthy",
            "Maintenance Priority": getattr(adv, "priority", "Low") if adv else "Low",
            "Recommended Action": getattr(adv, "recommended_action", "Routine review") if adv else "Routine review",
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # 4. Report Downloads & Export Center
    st.markdown("### Export Center & Report Downloads")
    col_pdf, col_csv, col_data = st.columns(3)

    from faultmesh.reporting import SENSOR_SUBSYSTEMS, generate_pdf_report, export_summary_csv

    # 1. PDF Report Download
    pdf_buffer = io.BytesIO()
    try:
        generate_pdf_report(summary, pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()
        col_pdf.download_button(
            label="Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"faultmesh_report_{r_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            help="High-density industrial engineering PDF report with complete findings.",
        )
    except Exception as exc:
        col_pdf.button("PDF Generation Error", disabled=True, use_container_width=True)
        st.warning(f"Could not compile PDF: {exc}")

    # 2. Diagnostic Summary CSV Download
    csv_buffer = io.StringIO()
    try:
        export_summary_csv(summary, csv_buffer)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        col_csv.download_button(
            label="Download Summary (CSV)",
            data=csv_bytes,
            file_name=f"faultmesh_summary_{r_id}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Flattened diagnostic telemetry, health, and maintenance priority CSV.",
        )
    except Exception as exc:
        col_csv.button("CSV Export Error", disabled=True, use_container_width=True)

    # 3. Raw Readings CSV Download
    if readings is not None and not readings.empty:
        readings_csv = readings.to_csv(index=False).encode("utf-8")
        col_data.download_button(
            label="Download Raw Telemetry (CSV)",
            data=readings_csv,
            file_name="sensor_readings.csv",
            mime="text/csv",
            use_container_width=True,
            help="Raw multi-sensor simulated telemetry time-series dataset.",
        )

    st.divider()

    # 5. Report History from SQLite Database
    _render_sqlite_report_history(st, db_path)


def _render_sqlite_report_history(st: Any, db_path: str = "data/faultmesh.db") -> None:
    """Safely query and display historical investigation reports from SQLite."""
    with st.expander("Historical Investigation Reports Database (SQLite)", expanded=False):
        import sqlite3
        import json

        if not os.path.isfile(db_path):
            st.caption(f"SQLite database not initialized at '{db_path}'. It will be created on simulation run.")
            return

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, created_at, report_json FROM fault_reports ORDER BY id DESC LIMIT 10"
                )
                rows = cursor.fetchall()

            if not rows:
                st.caption("No historical investigation reports stored in database yet.")
                return

            history_rows = []
            for row in rows:
                rep_id, created_at, json_str = row
                try:
                    data = json.loads(json_str)
                    loc = data.get("localization", {})
                    suspect = loc.get("primary_suspect", "N/A")
                    c_score = loc.get("composite_score", 0.0)
                    r_code = data.get("report_id", f"RPT-{rep_id}")
                    h_score = data.get("sensor_health", {}).get("network_health_score", 100.0)
                    attn = data.get("maintenance_summary", {}).get("overall_attention_level", "Nominal")
                except Exception:
                    suspect, c_score, r_code, h_score, attn = "N/A", 0.0, f"RPT-{rep_id}", 100.0, "N/A"

                history_rows.append({
                    "DB Record ID": rep_id,
                    "Report ID": r_code,
                    "Timestamp": created_at,
                    "Localized Suspect": suspect,
                    "Composite Score": f"{c_score:.4f}",
                    "Fleet Health": f"{h_score:.1f} / 100",
                    "Attention Level": attn,
                })

            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
            st.caption(f"Displaying {len(history_rows)} most recent diagnostic runs recorded in SQLite.")
        except Exception as exc:
            st.caption(f"Could not read historical reports from SQLite: {exc}")


# Backward compatibility aliases
render_investigation_reports_tab = render_executive_reports_tab
render_investigation_reports = render_executive_reports_tab


# -----------------------------------------------------------------------------
# Live IoT Mode Components
# -----------------------------------------------------------------------------

def render_live_comm_stats(
    st: Any,
    stats: Dict[str, Any],
    mqtt_status: Dict[str, Any],
) -> None:
    """Render comprehensive MQTT connection, subscription, and message validation metrics."""
    c1, c2, c3, c4 = st.columns(4)

    state = mqtt_status.get("connection_state", "DISCONNECTED")
    is_conn = mqtt_status.get("is_connected", False)
    sub_status = mqtt_status.get("subscription_status", "Inactive")
    is_sub = mqtt_status.get("is_subscribed", False)

    c1.metric(
        label="Broker Connection",
        value=state,
        delta=f"Sub: {sub_status}" if is_conn else "Offline",
        delta_color="normal" if is_conn and is_sub else ("off" if is_conn else "inverse"),
    )

    callbacks_cnt = mqtt_status.get("callbacks_received", 0)
    c2.metric(
        label="MQTT Callbacks Received",
        value=f"{callbacks_cnt:,} packets",
        delta=f"Topic: {mqtt_status.get('last_received_topic', 'None')}",
    )

    valid_cnt = stats.get("valid_count", 0)
    drained_cnt = mqtt_status.get("messages_drained", 0)
    c3.metric(
        label="Valid Telemetry Processed",
        value=f"{valid_cnt:,}",
        delta=f"{drained_cnt:,} drained from queue",
    )

    rej_cnt = stats.get("rejected_count", 0)
    q_size = mqtt_status.get("pending_queue_size", 0)
    c4.metric(
        label="Rejected / In Queue",
        value=f"{rej_cnt:,} rejected",
        delta=f"{q_size} in queue",
        delta_color="inverse" if rej_cnt > 0 else "normal",
    )

    if rej_cnt > 0 and stats.get("last_error"):
        st.warning(
            f"**Validation Rejection Diagnostic:** {rej_cnt} telemetry packet(s) rejected.\n\n"
            f"- **Latest Error:** `{stats.get('last_error')}`\n"
            f"- **Topic:** `{stats.get('last_rejected_topic', 'N/A')}`\n"
            f"- **Payload Snippet:** `{stats.get('last_rejected_payload', '<empty>')[:100]}`\n\n"
            "_Inspect Tab 'MQTT & Ingestion Logs' for the full diagnostic breakdown and recent rejection history._"
        )


def render_live_sensor_cards(
    st: Any,
    latest_readings: Dict[str, Dict[str, Any]],
) -> None:
    """Render 5 real-time KPI cards for live sensor observations."""
    cols = st.columns(5)
    from faultmesh.ingestion.data_validator import SUPPORTED_SENSORS

    labels = {
        "S1": "S1: Temperature",
        "S2": "S2: Vibration",
        "S3": "S3: Pressure",
        "S4": "S4: Flow",
        "S5": "S5: Current",
    }

    for idx, s in enumerate(SUPPORTED_SENSORS):
        info = latest_readings.get(s, {})
        val = info.get("value")
        unit = info.get("unit", "")
        stale = info.get("stale", True)
        status = info.get("status", "Offline")

        with cols[idx]:
            if val is not None:
                val_str = f"{val:.2f} {unit}"
                delta_str = "Live Streaming" if not stale else "Stale Reading"
                delta_color = "normal" if not stale else "inverse"
            else:
                val_str = f"-- {unit}"
                delta_str = status
                delta_color = "off"

            st.metric(
                label=labels.get(s, s),
                value=val_str,
                delta=delta_str,
                delta_color=delta_color,
            )


def render_live_diagnostic_transparency_table(
    st: Any,
    sensor_diagnostics: Dict[str, Dict[str, Any]],
) -> None:
    """Render comprehensive diagnostic transparency table for all five industrial sensors."""
    if not sensor_diagnostics:
        return

    st.markdown("#### Real-Time Sensor Diagnostic Transparency")
    st.caption("Live health, calibration, anomaly state, and root-cause classification per sensor.")

    rows = []
    for s, diag in sensor_diagnostics.items():
        val = diag.get("latest_value")
        unit = diag.get("unit", "")
        val_str = f"{val:.2f} {unit}" if val is not None else "No data"
        rows.append({
            "Sensor": f"{s} ({diag.get('name', s)})",
            "Latest Value": val_str,
            "Timestamp": diag.get("timestamp", "N/A"),
            "Valid Readings": diag.get("valid_count", 0),
            "Rejected": diag.get("rejected_count", 0),
            "Calibration": diag.get("calibration_status", "Uncalibrated"),
            "Sufficiency": diag.get("data_sufficiency", "Pending"),
            "Anomaly Score": f"{diag.get('anomaly_score', 0.0):.3f}",
            "Anomaly Status": diag.get("anomaly_status", "Normal"),
            "Health Score": f"{diag.get('health_score', 100.0):.1f} ({diag.get('health_status', 'Healthy')})",
            "Root-Cause State": diag.get("root_cause_status", "Normal"),
            "Diagnostic Reason": diag.get("diagnostic_reason", ""),
        })

    df_diag = pd.DataFrame(rows)
    st.dataframe(df_diag, use_container_width=True, hide_index=True)


def render_calibration_panel(
    st: Any,
    telem_mgr: Any,
) -> None:
    """Render baseline calibration workflow status and operator confirmation controls."""
    st.markdown("#### Baseline Calibration Workflow")
    st.markdown(
        "> **Calibration Requirement:** Industrial anomaly detection requires a confirmed normal baseline. "
        "The model does NOT automatically train on arbitrary live streams to prevent learning existing faults. "
        "An authorized operator must confirm that the system is operating nominally before calibrating."
    )

    can_cal, available, msg = telem_mgr.can_calibrate()
    target = telem_mgr.target_calibration_samples
    progress_val = min(1.0, float(available) / float(target)) if target > 0 else 0.0

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Baseline Progress:** {available} / {target} synchronized samples collected")
        st.progress(progress_val)
        if telem_mgr.calibration_status == "calibrated":
            st.success(
                f"**Baseline Calibrated** at `{telem_mgr.calibrated_at}` on {telem_mgr.calibration_samples_collected} samples. "
                "AI diagnostic engines active."
            )
        else:
            st.warning(f"**Calibration Pending:** {msg}")

    with col2:
        if telem_mgr.calibration_status == "calibrated":
            if st.button("Reset Calibration", use_container_width=True, type="secondary"):
                telem_mgr.reset_calibration()
                st.rerun()
        else:
            if st.button(
                "Confirm & Calibrate",
                use_container_width=True,
                type="primary",
                disabled=not can_cal,
                help="Confirm current operating conditions represent verified normal baseline and train Isolation Forest detector.",
            ):
                ok, c_msg = telem_mgr.confirm_and_calibrate()
                if ok:
                    st.success(c_msg)
                    st.rerun()
                else:
                    st.error(c_msg)


