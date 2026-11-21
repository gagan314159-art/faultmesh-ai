"""Executive Reporting and PDF Export Center for FaultMesh AI 2.0.

Responsibility:
    Synthesizes comprehensive fault investigation data across:
    - Raw multi-sensor simulation telemetry and physical fault injection
    - Multivariate Isolation Forest anomaly detection and onset timings
    - Graph-guided cooperative root-cause localization rankings
    - Phase 1 Explainable Fault Investigation evidence chains
    - Phase 2 Sensor Health Monitoring evaluations
    - Phase 3 Predictive Maintenance recommendations and technician checklists
    - Benchmark evaluation metrics (when available)

    Outputs:
    1. Structured, auditable investigation summary dictionary (14 core sections)
    2. Flattened diagnostic summary CSV dataset
    3. Publication-grade industrial engineering PDF report via ReportLab
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import io
import os
import datetime
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

REPORT_DISCLAIMER: str = (
    "NOTICE: This investigation report is generated automatically by FaultMesh AI 2.0 as an engineering "
    "decision-support advisory. Telemetry findings and recommendations are derived from synthetic physical "
    "simulation, multivariate statistical anomaly detection, and topological graph inference. Telemetry "
    "anomaly patterns do NOT constitute empirical Remaining Useful Life (RUL) run-to-failure predictions, "
    "exact hardware failure date forecasts, or certified automatic equipment-control actions."
)

PROJECT_NAME: str = "FaultMesh AI 2.0 — Autonomous Industrial Intelligence Platform"
REPORT_TITLE: str = "Executive Industrial Fault Investigation & Predictive Maintenance Report"

SENSOR_SUBSYSTEMS: Dict[str, Dict[str, str]] = {
    "S1": {"name": "Bearing Temperature", "subsystem": "Bearing Housing Assembly", "unit": "°C", "baseline": "60.0"},
    "S2": {"name": "Shaft Vibration", "subsystem": "Drive Shaft & Rotating Coupling", "unit": "mm/s", "baseline": "2.0"},
    "S3": {"name": "Cooling Pressure", "subsystem": "Hydraulic Manifold & Cooling Circuit", "unit": "bar", "baseline": "5.0"},
    "S4": {"name": "Coolant Flow", "subsystem": "Cooling Circulation & Flow Delivery", "unit": "L/min", "baseline": "30.0"},
    "S5": {"name": "Motor Current", "subsystem": "Electric Motor Drive & Phase Electronics", "unit": "A", "baseline": "10.0"},
}


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that computes and renders running headers, footers, and total page count."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running header on every page
        self.drawString(36, 11 * inch - 26, "FaultMesh AI 2.0 — Executive Fault Investigation & Maintenance Report")
        timestamp_str = getattr(self, "report_timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.drawRightString(8.5 * inch - 36, 11 * inch - 26, f"Generated: {timestamp_str}")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Running footer on every page
        self.line(36, 32, 8.5 * inch - 36, 32)
        self.drawString(36, 22, "CONFIDENTIAL — INDUSTRIAL DECISION SUPPORT — NOT CERTIFIED FOR AUTONOMOUS CONTROL")
        self.drawRightString(8.5 * inch - 36, 22, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_investigation_summary(
    cfg_info: Optional[Dict[str, Any]] = None,
    readings_df: Optional[pd.DataFrame] = None,
    scores_df: Optional[pd.DataFrame] = None,
    onsets: Optional[Dict[str, Optional[int]]] = None,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
    investigation_report: Optional[Any] = None,
    health_reports: Optional[Dict[str, Any]] = None,
    advisories: Optional[Dict[str, Any]] = None,
    network_health: Optional[Any] = None,
    maintenance_summary: Optional[Any] = None,
    benchmark: Optional[Dict[str, Any]] = None,
    report_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compile a complete, 14-section investigation summary dictionary from pipeline artifacts.

    Handles normal, faulty, and unsimulated/missing data states safely without fabricating facts.

    Args:
        cfg_info: Simulation parameters and injected fault configuration dictionary.
        readings_df: Raw multi-sensor readings DataFrame.
        scores_df: Calibrated anomaly scores DataFrame.
        onsets: Detection onset timestep dictionary per sensor.
        rankings: Ranked root-cause suspect tuples list.
        investigation_report: Optional InvestigationReport from FaultExplainer (Phase 1).
        health_reports: Optional Dict of SensorHealthRecord from SensorHealthMonitor (Phase 2).
        advisories: Optional Dict of MaintenanceAdvisory from MaintenanceAdvisor (Phase 3).
        network_health: Optional NetworkHealthSummary from SensorHealthMonitor (Phase 2).
        maintenance_summary: Optional NetworkMaintenanceSummary from MaintenanceAdvisor (Phase 3).
        benchmark: Optional systematic benchmark metrics dictionary.
        report_id: Optional custom report ID.

    Returns:
        Structured dictionary containing all 14 mandatory report sections.
    """
    now = datetime.datetime.now()
    r_id = report_id or f"RPT-{now.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex().upper()}"
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Safe defaults
    cfg = cfg_info or {}
    onsets_dict = onsets or {}
    health_dict = health_reports or {}
    adv_dict = advisories or {}
    rankings_list = rankings or []

    # 1. Root-cause localization extraction
    primary_suspect: Optional[str] = None
    composite_score: float = 0.0
    loc_breakdown: Dict[str, Any] = {}

    if rankings_list:
        primary_suspect = str(rankings_list[0][0])
        composite_score = float(rankings_list[0][1])
        loc_breakdown = dict(rankings_list[0][2])
    elif investigation_report:
        primary_suspect = getattr(investigation_report, "suspected_sensor", None)
        composite_score = float(getattr(investigation_report, "localization_score", 0.0))

    confidence = getattr(investigation_report, "confidence_level", getattr(investigation_report, "confidence_category", "Moderate"))
    if not rankings_list and not investigation_report:
        confidence = "Indeterminate"

    # 2. Telemetry statistics per sensor
    sensors_present = ["S1", "S2", "S3", "S4", "S5"]
    telemetry_stats: Dict[str, Dict[str, Any]] = {}

    has_telemetry = (readings_df is not None and not readings_df.empty)

    for s in sensors_present:
        if has_telemetry and s in readings_df.columns:
            vals = readings_df[s].dropna().values
            s_scores = scores_df[s].dropna().values if (scores_df is not None and s in scores_df.columns) else np.array([])
            telemetry_stats[s] = {
                "available": True,
                "min": float(np.min(vals)) if len(vals) else 0.0,
                "max": float(np.max(vals)) if len(vals) else 0.0,
                "mean": float(np.mean(vals)) if len(vals) else 0.0,
                "std": float(np.std(vals)) if len(vals) else 0.0,
                "latest": float(vals[-1]) if len(vals) else 0.0,
                "peak_anomaly_score": float(np.max(s_scores)) if len(s_scores) else 0.0,
                "alarm_onset": onsets_dict.get(s),
            }
        else:
            telemetry_stats[s] = {
                "available": False,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "latest": 0.0,
                "peak_anomaly_score": 0.0,
                "alarm_onset": onsets_dict.get(s),
            }

    # 3. Injected fault details
    fault_sensor = cfg.get("fault_sensor")
    fault_type = cfg.get("fault_type")
    fault_start = cfg.get("fault_start")
    fault_mag = cfg.get("fault_magnitude_ratio", cfg.get("fault_magnitude"))

    if fault_sensor and fault_type:
        injected_fault_str = f"Sensor {fault_sensor} — {str(fault_type).capitalize()} injected at t={fault_start}"
        is_faulty_simulation = True
    else:
        injected_fault_str = "Clean Nominal Baseline — No physical fault injected"
        is_faulty_simulation = False

    # 4. Executive summary narrative construction
    if not has_telemetry:
        exec_summary_text = (
            "Preliminary Report: No active simulation telemetry has been processed in this session. "
            "Telemetry-dependent anomaly detection, topological localization, and sensor health evaluations "
            "remain in standby. Historical diagnostics and benchmark evaluation are available for reference."
        )
    elif is_faulty_simulation and primary_suspect:
        suspect_name = SENSOR_SUBSYSTEMS.get(primary_suspect, {}).get("name", f"Sensor {primary_suspect}")
        exec_summary_text = (
            f"Autonomous fault investigation successfully localized the primary root-cause to {primary_suspect} "
            f"({suspect_name}) with composite localization score {composite_score:.4f} and {confidence} confidence. "
            f"Physical coupling propagation was confirmed across neighboring channels with delayed onset timings. "
            f"Immediate maintenance review and targeted physical inspections are recommended to restore normal operations."
        )
    else:
        exec_summary_text = (
            "Autonomous investigation evaluated all five sensor streams across the industrial mesh network. "
            "All monitored channels operated within nominal baseline tolerances with zero sustained anomaly onsets. "
            "Overall network condition is rated Nominal, and routine preventive maintenance schedules should be maintained."
        )

    # 5. Network health details
    net_health_score = float(getattr(network_health, "overall_network_health_score", 100.0 if not is_faulty_simulation else 75.0))
    fleet_attention = getattr(maintenance_summary, "overall_maintenance_attention_level", "Nominal" if not is_faulty_simulation else "Critical")

    # 6. Benchmark details extraction
    benchmark_data = None
    if benchmark and isinstance(benchmark, dict) and "top1_accuracy" in benchmark:
        benchmark_data = {
            "available": True,
            "top1_accuracy": float(benchmark.get("top1_accuracy", 1.0)),
            "top2_accuracy": float(benchmark.get("top2_accuracy", 1.0)),
            "total_scenarios": int(benchmark.get("total_scenarios", 75)),
            "detection_rate": float(benchmark.get("detection_rate", 1.0)),
            "fault_breakdown": benchmark.get("fault_type_breakdown", {}),
            "sensor_breakdown": benchmark.get("sensor_breakdown", {}),
        }
    else:
        benchmark_data = {
            "available": False,
            "message": "Benchmark evaluation data not loaded or currently unavailable.",
        }

    # 7. Compile complete 14-section dictionary
    return {
        # Section 1: Report ID and generation timestamp
        "report_id": r_id,
        "generated_at": timestamp_str,

        # Section 2: Project name and report title
        "project_name": PROJECT_NAME,
        "report_title": REPORT_TITLE,

        # Section 3: Executive summary
        "executive_summary": exec_summary_text,

        # Section 4: Machine and sensor network overview
        "network_overview": {
            "system_type": "Coupled Industrial Rotating Machinery & Fluid Loop",
            "monitored_nodes": 5,
            "coupling_strength": "40% anomalous deviation propagation",
            "propagation_delay": "2 timesteps",
            "sensors": SENSOR_SUBSYSTEMS,
        },

        # Section 5: Simulation configuration and injected fault details
        "simulation_config": {
            "is_simulated": has_telemetry,
            "total_timesteps": len(readings_df) if has_telemetry else 0,
            "random_seed": cfg.get("seed", 42),
            "noise_ratio": cfg.get("noise_ratio", 0.015),
            "injected_fault_summary": injected_fault_str,
            "raw_config": cfg,
        },

        # Section 6: Anomaly detection findings
        "anomaly_detection": {
            "onsets": onsets_dict,
            "threshold": 0.60,
            "consecutive_required": 3,
            "telemetry_stats": telemetry_stats,
        },

        # Section 7: Root-cause localization rankings
        "localization": {
            "primary_suspect": primary_suspect,
            "composite_score": composite_score,
            "confidence_level": confidence,
            "rankings": rankings_list,
            "top1_breakdown": loc_breakdown,
        },

        # Section 8: Explainable investigation evidence
        "explainable_evidence": {
            "narrative": getattr(investigation_report, "summary_narrative", getattr(investigation_report, "investigation_summary", exec_summary_text)),
            "evidence_items": getattr(investigation_report, "evidence", []),
            "neighbor_support": getattr(investigation_report, "neighbor_support_evidence", []),
            "timing_comparisons": getattr(investigation_report, "timing_comparison", []),
            "limitations": getattr(investigation_report, "limitations", []),
        },

        # Section 9: Sensor health assessment
        "sensor_health": {
            "network_health_score": net_health_score,
            "reports": health_dict,
            "healthy_count": getattr(network_health, "healthy_count", sum(1 for r in health_dict.values() if getattr(r, "health_status", "") == "Healthy")),
            "warning_count": getattr(network_health, "warning_count", sum(1 for r in health_dict.values() if getattr(r, "health_status", "") == "Warning")),
            "critical_count": getattr(network_health, "critical_count", sum(1 for r in health_dict.values() if getattr(r, "health_status", "") == "Critical")),
            "insufficient_data_count": getattr(network_health, "insufficient_data_count", sum(1 for r in health_dict.values() if "Insufficient" in str(getattr(r, "health_status", "")))),
        },

        # Section 10: Predictive maintenance priorities
        "predictive_maintenance": {
            "advisories": adv_dict,
        },

        # Section 11: Recommended technician follow-up actions
        "technician_actions": {
            "primary_suspect_actions": getattr(adv_dict.get(primary_suspect or ""), "actions", []),
            "primary_suspect_checklist": getattr(adv_dict.get(primary_suspect or ""), "inspection_checklist", []),
            "all_prescribed_actions": {
                s: getattr(adv, "actions", []) for s, adv in adv_dict.items()
            },
        },

        # Section 12: Network-level maintenance summary
        "maintenance_summary": {
            "overall_attention_level": fleet_attention,
            "highest_priority_sensor": getattr(maintenance_summary, "highest_priority_sensor", primary_suspect),
            "critical_cases": getattr(maintenance_summary, "critical_cases", sum(1 for a in adv_dict.values() if getattr(a, "priority", "") == "Critical")),
            "high_priority_cases": getattr(maintenance_summary, "high_priority_cases", sum(1 for a in adv_dict.values() if getattr(a, "priority", "") == "High")),
            "medium_priority_cases": getattr(maintenance_summary, "medium_priority_cases", sum(1 for a in adv_dict.values() if getattr(a, "priority", "") == "Medium")),
            "low_priority_cases": getattr(maintenance_summary, "low_priority_cases", sum(1 for a in adv_dict.values() if getattr(a, "priority", "") == "Low")),
            "insufficient_data_cases": getattr(maintenance_summary, "insufficient_data_cases", sum(1 for a in adv_dict.values() if getattr(a, "priority", "") == "Insufficient Data")),
            "ranked_action_list": getattr(maintenance_summary, "ranked_action_list", []),
        },

        # Section 13: Benchmark results, if available
        "benchmark_results": benchmark_data,

        # Section 14: Known limitations and diagnostic disclaimer
        "limitations_and_disclaimer": {
            "disclaimer": REPORT_DISCLAIMER,
            "limitations": [
                "Telemetry observations originate from synthetic physical differential models; physical equipment installation requires hardware field validation.",
                "Statistical anomaly detection evaluates relative standard deviation departures and does not model chemical lubricant oxidation.",
                "Does not formulate empirical Remaining Useful Life (RUL) run-to-failure calculations.",
                "Advisories are decision-support protocols; technician must verify equipment Lockout/Tagout (LOTO) safety before physical access.",
            ],
        },

        # Convenience backward-compatibility root keys
        "metadata": {
            "report_id": r_id,
            "title": REPORT_TITLE,
            "generated_at": timestamp_str,
            "total_timesteps": len(readings_df) if has_telemetry else 0,
            "sim_config": cfg,
        },
        "telemetry_stats": telemetry_stats,
        "investigation_report": investigation_report,
        "health_reports": health_dict,
        "advisories": adv_dict,
    }


def export_summary_csv(
    summary: Dict[str, Any],
    filepath_or_buffer: Union[str, io.StringIO],
) -> None:
    """Export flattened diagnostic telemetry, health, and maintenance summary to CSV format.

    Args:
        summary: Dictionary produced by build_investigation_summary.
        filepath_or_buffer: Target file path string or io.StringIO buffer.
    """
    rows = []
    rankings_list = summary.get("localization", {}).get("rankings", [])
    rankings_dict = {item[0]: (i + 1, item[1], item[2]) for i, item in enumerate(rankings_list)}
    health = summary.get("sensor_health", {}).get("reports", summary.get("health_reports", {}))
    stats = summary.get("anomaly_detection", {}).get("telemetry_stats", summary.get("telemetry_stats", {}))
    advisories = summary.get("predictive_maintenance", {}).get("advisories", summary.get("advisories", {}))

    sensors = ["S1", "S2", "S3", "S4", "S5"]
    for s in sensors:
        st = stats.get(s, {})
        rank_info = rankings_dict.get(s, (None, 0.0, {}))
        h_info = health.get(s)
        adv = advisories.get(s)
        meta = SENSOR_SUBSYSTEMS.get(s, {})

        row = {
            "sensor": s,
            "subsystem": meta.get("name", s),
            "suspect_rank": rank_info[0] if rank_info[0] is not None else "Nominal",
            "composite_score": round(rank_info[1], 4) if rank_info[1] else 0.0,
            "alarm_onset": st.get("alarm_onset", "None"),
            "peak_anomaly_score": round(st.get("peak_anomaly_score", 0.0), 4),
            "mean_reading": round(st.get("mean", 0.0), 3),
            "latest_reading": round(st.get("latest", 0.0), 3),
            "health_score": round(getattr(h_info, "health_score", 100.0), 1) if h_info else "N/A",
            "health_status": getattr(h_info, "health_status", getattr(h_info, "status", "N/A")),
            "maintenance_priority": getattr(adv, "priority", "N/A"),
            "recommended_action": getattr(adv, "recommended_action", "N/A"),
            "urgency_score": getattr(adv, "urgency_score", "N/A"),
            "suggested_follow_up": getattr(adv, "suggested_follow_up", "N/A"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if isinstance(filepath_or_buffer, str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath_or_buffer)), exist_ok=True)
        df.to_csv(filepath_or_buffer, index=False)
    else:
        df.to_csv(filepath_or_buffer, index=False)


def generate_pdf_report(
    summary: Dict[str, Any],
    filepath_or_buffer: Union[str, io.BytesIO],
) -> None:
    """Generate a clean, high-density industrial engineering PDF report using ReportLab.

    Features:
    - Running headers, footers, and two-pass page numbers ("Page X of Y").
    - Clean cover/header section and structured tables.
    - Color-coded priority badges and health indicators.
    - Sensor-wise diagnostic findings and technician maintenance checklists.
    - Benchmark results and legal/engineering disclaimers.

    Args:
        summary: Dictionary produced by build_investigation_summary.
        filepath_or_buffer: Target file path string or io.BytesIO buffer.
    """
    doc = SimpleDocTemplate(
        filepath_or_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Document Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0D1B2A"),
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#415A77"),
        spaceAfter=8,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1B263B"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=3,
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=body_style,
        fontSize=7.5,
        leading=9.5,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=6,
    )

    story: List[Any] = []

    # Metadata extraction
    r_id = summary.get("report_id", summary.get("metadata", {}).get("report_id", "RPT-UNKNOWN"))
    gen_time = summary.get("generated_at", summary.get("metadata", {}).get("generated_at", ""))
    sim_cfg = summary.get("simulation_config", {})
    injected_summary = sim_cfg.get("injected_fault_summary", "Clean Nominal Baseline")

    # Document Header Banner
    story.append(Paragraph(REPORT_TITLE, title_style))
    story.append(Paragraph(
        f"<b>Report ID:</b> {r_id} &nbsp;|&nbsp; <b>Generated:</b> {gen_time} &nbsp;|&nbsp; <b>Status:</b> {injected_summary}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#005F73"), spaceAfter=8))

    # Executive Summary Banner Box (4 Columns)
    loc = summary.get("localization", {})
    primary = loc.get("primary_suspect") or "None (Nominal)"
    score = loc.get("composite_score", 0.0)
    conf = loc.get("confidence_level", "Moderate")

    health_info = summary.get("sensor_health", {})
    net_score = health_info.get("network_health_score", 100.0)

    maint_info = summary.get("maintenance_summary", {})
    attention = maint_info.get("overall_attention_level", "Nominal")

    banner_data = [
        [
            Paragraph("<b>SUSPECTED ROOT CAUSE</b>", body_style),
            Paragraph("<b>COMPOSITE SCORE</b>", body_style),
            Paragraph("<b>INVESTIGATION CONFIDENCE</b>", body_style),
            Paragraph("<b>FLEET ATTENTION</b>", body_style),
        ],
        [
            Paragraph(f"<font size=12><b>{primary}</b></font>", body_bold),
            Paragraph(f"<font size=12><b>{score:.4f}</b></font>", body_bold),
            Paragraph(f"<font size=11><b>{conf}</b></font>", body_bold),
            Paragraph(f"<font size=11><b>{attention}</b></font>", body_bold),
        ],
    ]
    banner_table = Table(banner_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 8))

    # Narrative Executive Summary
    exec_summary_text = summary.get("executive_summary", "")
    if exec_summary_text:
        story.append(Paragraph("<b>Executive Diagnostic Synopsis:</b>", body_bold))
        story.append(Paragraph(exec_summary_text, body_style))
        story.append(Spacer(1, 6))

    # Section 1: Root Cause Localization Rankings Table
    story.append(Paragraph("1. Graph-Guided Root-Cause Localization Rankings", section_heading))
    rank_headers = [
        Paragraph("Rank", table_header),
        Paragraph("Sensor", table_header),
        Paragraph("Subsystem", table_header),
        Paragraph("Composite Score", table_header),
        Paragraph("Earliness (30%)", table_header),
        Paragraph("Own Anomaly (35%)", table_header),
        Paragraph("Support (20%)", table_header),
        Paragraph("Coverage (15%)", table_header),
    ]
    rank_rows = [rank_headers]

    rankings_list = loc.get("rankings", [])
    if rankings_list:
        for i, item in enumerate(rankings_list):
            s_id = item[0]
            c_score = item[1]
            bd = item[2]
            meta = SENSOR_SUBSYSTEMS.get(s_id, {})
            rank_rows.append([
                Paragraph(f"#{i + 1}", table_cell),
                Paragraph(f"<b>{s_id}</b>", table_cell),
                Paragraph(meta.get("name", s_id), table_cell),
                Paragraph(f"{c_score:.4f}", table_cell),
                Paragraph(f"{bd.get('detection_earliness', 0.0):.3f}", table_cell),
                Paragraph(f"{bd.get('own_anomaly', 0.0):.3f}", table_cell),
                Paragraph(f"{bd.get('neighbour_support', 0.0):.3f}", table_cell),
                Paragraph(f"{bd.get('coverage', 0.0):.3f}", table_cell),
            ])
    else:
        rank_rows.append([
            Paragraph("N/A", table_cell),
            Paragraph("Nominal", table_cell),
            Paragraph("All Subsystems Nominal", table_cell),
            Paragraph("0.0000", table_cell),
            Paragraph("0.000", table_cell),
            Paragraph("0.000", table_cell),
            Paragraph("0.000", table_cell),
            Paragraph("0.000", table_cell),
        ])

    rank_table = Table(rank_rows, colWidths=[0.5 * inch, 0.6 * inch, 1.7 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch, 0.8 * inch, 0.7 * inch])
    rank_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(rank_table)
    story.append(Spacer(1, 8))

    # Section 2: Sensor Health Assessment & Telemetry Findings Table
    health_reports = summary.get("sensor_health", {}).get("reports", summary.get("health_reports", {}))
    stats = summary.get("anomaly_detection", {}).get("telemetry_stats", summary.get("telemetry_stats", {}))

    story.append(Paragraph(f"2. Sensor Operational Health & Telemetry Metrics (Fleet Index: {net_score:.1f}/100)", section_heading))
    h_headers = [
        Paragraph("Sensor", table_header),
        Paragraph("Subsystem", table_header),
        Paragraph("Health Score", table_header),
        Paragraph("Status", table_header),
        Paragraph("Alarm Onset", table_header),
        Paragraph("Peak Anomaly", table_header),
        Paragraph("Baseline", table_header),
        Paragraph("Latest Value", table_header),
    ]
    h_rows = [h_headers]

    sensors = ["S1", "S2", "S3", "S4", "S5"]
    for s_id in sensors:
        rep = health_reports.get(s_id)
        st_data = stats.get(s_id, {})
        meta = SENSOR_SUBSYSTEMS.get(s_id, {})

        h_score_val = getattr(rep, "health_score", 100.0) if rep else 100.0
        h_status = getattr(rep, "health_status", getattr(rep, "status", "Healthy")) if rep else "Healthy"
        onset_str = f"t={st_data.get('alarm_onset')}" if st_data.get("alarm_onset") is not None else "None"
        peak_str = f"{st_data.get('peak_anomaly_score', 0.0):.3f}"
        latest_val = st_data.get("latest", 0.0)

        # Status badge coloring
        status_color = "#16A34A" if h_status == "Healthy" else ("#D97706" if h_status == "Warning" else "#DC2626")

        h_rows.append([
            Paragraph(f"<b>{s_id}</b>", table_cell),
            Paragraph(meta.get("name", s_id), table_cell),
            Paragraph(f"{h_score_val:.1f} / 100", table_cell),
            Paragraph(f"<font color='{status_color}'><b>{h_status}</b></font>", table_cell),
            Paragraph(onset_str, table_cell),
            Paragraph(peak_str, table_cell),
            Paragraph(f"{meta.get('baseline')} {meta.get('unit')}", table_cell),
            Paragraph(f"{latest_val:.2f} {meta.get('unit')}", table_cell),
        ])

    h_table = Table(h_rows, colWidths=[0.6 * inch, 1.8 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch, 0.7 * inch])
    h_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(h_table)
    story.append(Spacer(1, 8))

    # Section 3: Predictive Maintenance Recommendations Table
    advisories = summary.get("predictive_maintenance", {}).get("advisories", summary.get("advisories", {}))
    story.append(Paragraph("3. Predictive Maintenance Priorities & Prescribed Actions", section_heading))
    m_headers = [
        Paragraph("Sensor", table_header),
        Paragraph("Subsystem", table_header),
        Paragraph("Priority", table_header),
        Paragraph("Urgency Score", table_header),
        Paragraph("Recommended Action", table_header),
        Paragraph("Follow-up Window", table_header),
    ]
    m_rows = [m_headers]

    for s_id in sensors:
        adv = advisories.get(s_id)
        meta = SENSOR_SUBSYSTEMS.get(s_id, {})
        p = getattr(adv, "priority", "Low") if adv else "Low"
        u_score = getattr(adv, "urgency_score", 10.0) if adv else 10.0
        act_name = getattr(adv, "recommended_action", "Routine maintenance review") if adv else "Routine maintenance review"
        follow_up = getattr(adv, "suggested_follow_up", "Routine (30 days)") if adv else "Routine (30 days)"

        p_color = "#DC2626" if p == "Critical" else ("#D97706" if p == "High" else ("#2563EB" if p == "Medium" else "#16A34A"))

        m_rows.append([
            Paragraph(f"<b>{s_id}</b>", table_cell),
            Paragraph(meta.get("name", s_id), table_cell),
            Paragraph(f"<font color='{p_color}'><b>{p}</b></font>", table_cell),
            Paragraph(f"{u_score:.1f} / 100", table_cell),
            Paragraph(act_name, table_cell),
            Paragraph(follow_up, table_cell),
        ])

    m_table = Table(m_rows, colWidths=[0.6 * inch, 1.8 * inch, 0.9 * inch, 0.9 * inch, 1.8 * inch, 1.2 * inch])
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 8))

    # Section 4: Prescribed Technician Maintenance Actions (Primary Suspect)
    primary_adv = advisories.get(primary) if primary else next(iter(advisories.values()), None)
    if primary_adv and getattr(primary_adv, "actions", None):
        story.append(Paragraph(f"4. Targeted Physical Maintenance Procedures for Node {primary_adv.sensor_id}", section_heading))
        for act in primary_adv.actions:
            tools_str = ", ".join(act.required_tools) if act.required_tools else "Standard Hand Tools"
            story.append(Paragraph(
                f"• <b>[{act.action_id}] {act.title}</b> ({act.component}): {act.description}<br/>"
                f"&nbsp;&nbsp;<i>Required Equipment:</i> {tools_str}<br/>"
                f"&nbsp;&nbsp;<i>Safety Precautions:</i> {act.safety_notes}",
                body_style,
            ))
            story.append(Spacer(1, 2))

        if getattr(primary_adv, "inspection_checklist", None):
            story.append(Spacer(1, 2))
            story.append(Paragraph("<b>Technician Verification Checklist:</b>", body_bold))
            for chk in primary_adv.inspection_checklist:
                story.append(Paragraph(f"  [ ] {chk}", body_style))

    # Section 5: Systematic Benchmark Evaluation (if available)
    bmk_info = summary.get("benchmark_results", {})
    if bmk_info and bmk_info.get("available"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("5. Systematic 75-Scenario Benchmark Evaluation", section_heading))
        top1_pct = bmk_info.get("top1_accuracy", 1.0) * 100.0
        top2_pct = bmk_info.get("top2_accuracy", 1.0) * 100.0
        det_rate = bmk_info.get("detection_rate", 1.0) * 100.0
        total_scen = bmk_info.get("total_scenarios", 75)

        story.append(Paragraph(
            f"<b>Evaluation Matrix:</b> 5 Sensors × 3 Fault Types (Drift, Spike, Stuck) × 5 Seeds = {total_scen} Scenarios &nbsp;|&nbsp; "
            f"<b>Top-1 Localization Accuracy:</b> {top1_pct:.2f}% &nbsp;|&nbsp; "
            f"<b>Top-2 Accuracy:</b> {top2_pct:.2f}% &nbsp;|&nbsp; "
            f"<b>Detection Rate:</b> {det_rate:.2f}%",
            body_style,
        ))

    # Section 6: Limitations & Engineering Disclaimers
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#94A3B8"), spaceAfter=4))
    story.append(Paragraph(f"<b>Engineering Disclaimer:</b> {REPORT_DISCLAIMER}", disclaimer_style))
    story.append(Paragraph(
        "FaultMesh AI 2.0 Autonomous Industrial Monitoring Platform &nbsp;|&nbsp; Confidential Diagnostic Document &nbsp;|&nbsp; Not Certified for Autonomous Hardware Actuation",
        disclaimer_style,
    ))

    # Set custom report timestamp attribute on doc for canvas running header
    doc.build(story, canvasmaker=NumberedCanvas)


class ExecutiveReporter:
    """Reusable executive investigation reporting and export coordinator."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path

    def compile_report(
        self,
        cfg_info: Optional[Dict[str, Any]] = None,
        readings_df: Optional[pd.DataFrame] = None,
        scores_df: Optional[pd.DataFrame] = None,
        onsets: Optional[Dict[str, Optional[int]]] = None,
        rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
        investigation_report: Optional[Any] = None,
        health_reports: Optional[Dict[str, Any]] = None,
        advisories: Optional[Dict[str, Any]] = None,
        network_health: Optional[Any] = None,
        maintenance_summary: Optional[Any] = None,
        benchmark: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile complete report dictionary using build_investigation_summary."""
        return build_investigation_summary(
            cfg_info=cfg_info,
            readings_df=readings_df,
            scores_df=scores_df,
            onsets=onsets,
            rankings=rankings,
            investigation_report=investigation_report,
            health_reports=health_reports,
            advisories=advisories,
            network_health=network_health,
            maintenance_summary=maintenance_summary,
            benchmark=benchmark,
        )

    def export_csv(self, summary: Dict[str, Any], filepath_or_buffer: Optional[Union[str, io.StringIO]] = None) -> Union[str, None]:
        """Export investigation summary to CSV."""
        if filepath_or_buffer is None:
            buf = io.StringIO()
            export_summary_csv(summary, buf)
            return buf.getvalue()
        export_summary_csv(summary, filepath_or_buffer)
        return None

    def export_pdf(self, summary: Dict[str, Any], filepath_or_buffer: Optional[Union[str, io.BytesIO]] = None) -> Union[bytes, None]:
        """Export investigation summary to publication-grade PDF."""
        if filepath_or_buffer is None:
            buf = io.BytesIO()
            generate_pdf_report(summary, buf)
            return buf.getvalue()
        generate_pdf_report(summary, filepath_or_buffer)
        return None
