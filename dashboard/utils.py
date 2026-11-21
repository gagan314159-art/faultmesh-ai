"""Dashboard utility functions for FaultMesh AI.

Responsibility:
    Connects the Streamlit dashboard to verified FaultMesh AI backend modules:
    simulation, feature extraction, anomaly detection, graph-based localization,
    deterministic caching, and benchmark metric loading.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

from faultmesh.simulator import SensorSimulator, DEFAULT_BASELINES
from faultmesh.processing import add_features
from faultmesh.detector import AnomalyDetector, first_onset
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer
from faultmesh.explainer import FaultExplainer, InvestigationReport
from faultmesh.health import SensorHealthMonitor, SensorHealthReport, NetworkHealthSummary
from faultmesh.maintenance import MaintenanceAdvisor, MaintenanceAdvisory, NetworkMaintenanceSummary
from faultmesh.reporting import build_investigation_summary, export_summary_csv, generate_pdf_report
from evaluate import calculate_metrics

CACHE_DIR = Path("data/cache")
DEFAULT_BENCHMARK_CSV = "data/evaluation_results.csv"
SENSORS = ["S1", "S2", "S3", "S4", "S5"]


@dataclass
class SimConfig:
    """Simulation parameters configuration container."""
    seed: int = 42
    fault_sensor: str = "S3"
    fault_type: str = "drift"
    fault_magnitude_ratio: float = 0.5
    fault_start: int = 150
    spike_duration: int = 10
    total_steps: int = 300
    noise_ratio: float = 0.015
    coupling_strength: float = 0.40
    coupling_delay: int = 2


def _normalize_cfg(cfg: Any) -> Dict[str, Any]:
    """Convert dataclass, dict, or object into standard dictionary."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"raw": str(cfg)}


def cache_key(cfg: Any) -> str:
    """Generate a deterministic SHA-256 cache key from all simulation parameters.

    Args:
        cfg: Configuration dataclass, dictionary, or object.

    Returns:
        64-character hexadecimal SHA-256 string.
    """
    raw_data = _normalize_cfg(cfg)
    # Normalize float values to 6 decimal places for consistent cross-platform hashing
    normalized: Dict[str, Any] = {}
    for k, v in sorted(raw_data.items()):
        if isinstance(v, float):
            normalized[k] = round(v, 6)
        else:
            normalized[k] = v

    serialized = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_save(key: str, df: pd.DataFrame, fault: Optional[Dict[str, Any]] = None) -> None:
    """Save simulation DataFrame and fault metadata to disk cache.

    Args:
        key: SHA-256 cache key.
        df: Simulation readings DataFrame.
        fault: Ground-truth fault parameter dictionary or None.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{key}.pkl"
        temp_file = CACHE_DIR / f"{key}.tmp"

        df_to_save = df.copy()
        if fault is not None:
            df_to_save.attrs["fault"] = dict(fault)

        payload = {
            "df": df_to_save,
            "fault": dict(fault) if fault is not None else None,
        }

        with open(temp_file, "wb") as f:
            pickle.dump(payload, f)
        temp_file.replace(cache_file)
    except Exception:
        # Cache failures must never break execution
        pass


def cache_load(key: str) -> Optional[Tuple[pd.DataFrame, Optional[Dict[str, Any]]]]:
    """Safely load cached simulation DataFrame and fault metadata.

    Args:
        key: SHA-256 cache key.

    Returns:
        Tuple of (DataFrame, fault_dict) if valid cache hit, else None.
    """
    cache_file = CACHE_DIR / f"{key}.pkl"
    if not cache_file.is_file():
        return None

    try:
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "df" in payload:
            df = payload["df"]
            fault = payload.get("fault")
        elif isinstance(payload, tuple) and len(payload) == 2:
            df, fault = payload
        elif isinstance(payload, pd.DataFrame):
            df = payload
            fault = df.attrs.get("fault")
        else:
            return None

        if not isinstance(df, pd.DataFrame):
            return None

        # Re-attach fault metadata if available
        if fault is not None:
            df.attrs["fault"] = dict(fault)
        elif "fault" not in df.attrs:
            df.attrs["fault"] = None

        return df, fault
    except Exception:
        # Corrupted or unpicklable cache file handled safely
        return None


def _extract_sim_params(cfg: Any) -> Tuple[int, int, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Extract and validate parameters for SensorSimulator.generate()."""
    d = _normalize_cfg(cfg)

    seed = int(d.get("seed", 42))
    n_steps = int(d.get("total_steps", d.get("n_steps", 300)))
    sensor = d.get("fault_sensor", d.get("sensor", "S3"))
    fault_type = d.get("fault_type", d.get("type", "drift"))
    start = int(d.get("fault_start", d.get("start", 150)))

    # Simulator structural parameters
    sim_kwargs = {
        "noise_ratio": float(d.get("noise_ratio", 0.015)),
        "coupling_strength": float(d.get("coupling_strength", 0.40)),
        "coupling_delay": int(d.get("coupling_delay", 2)),
    }

    if sensor in (None, "None", "none", "") or fault_type in (None, "None", "none", ""):
        return seed, n_steps, None, sim_kwargs

    base = DEFAULT_BASELINES.get(sensor, 10.0)
    if "magnitude" in d and d["magnitude"] is not None:
        mag = float(d["magnitude"])
    elif "fault_magnitude_ratio" in d and d["fault_magnitude_ratio"] is not None:
        mag = float(base * float(d["fault_magnitude_ratio"]))
    elif "magnitude_ratio" in d and d["magnitude_ratio"] is not None:
        mag = float(base * float(d["magnitude_ratio"]))
    else:
        mag = float(base * 0.5)

    fault_dict: Dict[str, Any] = {
        "sensor": sensor,
        "type": fault_type,
        "start": start,
        "magnitude": mag,
    }

    duration = d.get("spike_duration", d.get("fault_duration", d.get("duration")))
    if duration is not None:
        fault_dict["duration"] = int(duration)
    elif fault_type == "spike":
        fault_dict["duration"] = 10

    return seed, n_steps, fault_dict, sim_kwargs


def run_simulation(cfg: Any, use_cache: bool = True) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Execute sensor network simulation with caching and fault metadata preservation.

    Args:
        cfg: Configuration dataclass or dictionary.
        use_cache: If True, attempts to retrieve or write to disk cache.

    Returns:
        Tuple of (simulated readings DataFrame, ground-truth fault dictionary or None).
    """
    key = cache_key(cfg)
    if use_cache:
        cached = cache_load(key)
        if cached is not None:
            df, fault = cached
            if fault is not None and df.attrs.get("fault") is None:
                df.attrs["fault"] = fault
            return df, fault

    seed, n_steps, fault_dict, sim_kwargs = _extract_sim_params(cfg)
    sim = SensorSimulator(seed=seed, **sim_kwargs)
    df = sim.generate(n_steps=n_steps, fault=fault_dict)

    if use_cache:
        cache_save(key, df, fault_dict)

    return df, fault_dict


def run_detection(
    df: pd.DataFrame,
    threshold: float = 0.6,
    consecutive_required: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Optional[int]]]:
    """Train detector strictly on normal samples and compute anomaly scores and onsets.

    Features are calculated via add_features(). The detector fits exclusively
    on the first 100 normal baseline samples (t=0..99).

    Args:
        df: Input DataFrame containing raw sensor readings.
        threshold: Score cutoff for alarm detection (default: 0.6).
        consecutive_required: Consecutive abnormal samples required to confirm onset (default: 3).

    Returns:
        Tuple of:
            - scores_df: DataFrame of calibrated anomaly scores in [0, 1] per sensor.
            - onsets: Dict mapping sensor ID to first detection onset timestep or None.
    """
    # 1. Feature engineering
    feats_df = add_features(df)

    # 2. Train detector strictly on the first 100 normal samples
    detector = AnomalyDetector(
        threshold=threshold,
        consecutive_required=consecutive_required,
        random_state=42,
    )
    detector.fit(feats_df.iloc[:100])

    # 3. Compute calibrated anomaly scores across all timesteps
    scores_df = detector.score(feats_df)

    # 4. Extract sustained detection onsets
    onsets = first_onset(scores_df, threshold=threshold, min_consecutive=consecutive_required)

    return scores_df, onsets


def run_localization(
    scores_df: pd.DataFrame,
    onsets: Dict[str, Optional[int]],
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Perform graph-guided root-cause localization without ground-truth leakage.

    Preserves verified weights: 35% Own, 30% Earliness, 20% Support, 15% Coverage.
    Anomaly score profiles are evaluated directly from scores_df: transient faults
    (spikes) evaluate peak scores, while sustained faults evaluate the mean of the
    final 50 steps.

    Args:
        scores_df: Calibrated anomaly scores DataFrame.
        onsets: Detection onset timestep dictionary per sensor.

    Returns:
        Ranked suspect tuples: (sensor_id, composite_score, breakdown_dict).
    """
    sensor_cols = [s for s in SENSORS if s in scores_df.columns]
    n_steps = len(scores_df)

    # Identify whether the anomaly event was transient or sustained directly from score profiles
    tail_slice = scores_df.iloc[-50:] if n_steps >= 50 else scores_df
    is_transient = any(
        float(scores_df[s].max()) >= 0.6 and float(tail_slice[s].mean()) < 0.4
        for s in sensor_cols
    )

    if is_transient:
        anomaly_scores = {s: float(scores_df[s].max()) for s in sensor_cols}
    else:
        anomaly_scores = {s: float(tail_slice[s].mean()) for s in sensor_cols}

    localizer = FaultLocalizer()
    rankings = localizer.rank_suspects(anomaly_scores, onsets, total_steps=n_steps)
    return rankings


def load_benchmark(
    csv_path: Union[str, Path] = DEFAULT_BENCHMARK_CSV,
) -> Optional[Dict[str, Any]]:
    """Safely load benchmark evaluation CSV and calculate evaluation metrics.

    Never hardcodes metrics. Reuses evaluate.calculate_metrics() on the actual
    scenario records.

    Args:
        csv_path: File path to evaluation_results.csv.

    Returns:
        Metrics dictionary containing top1_accuracy, top2_accuracy, breakdowns,
        and underlying DataFrame, or None if the file is missing or invalid.
    """
    path = Path(csv_path)
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            return None

        # Ensure boolean columns are parsed accurately
        bool_cols = ["top1_correct", "top2_correct", "detection_success"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1", "t"])

        records = df.to_dict(orient="records")
        metrics = calculate_metrics(records)
        metrics["df"] = df
        metrics["records"] = records
        metrics["csv_path"] = str(path)
        return metrics
    except Exception:
        return None


def run_explainability(
    scores_df: pd.DataFrame,
    onsets: Dict[str, Optional[int]],
    rankings: List[Tuple[str, float, Dict[str, Any]]],
    graph: Optional[SensorNetworkGraph] = None,
) -> InvestigationReport:
    """Execute explainable fault investigation decomposition."""
    explainer = FaultExplainer(network_graph=graph)
    return explainer.explain(scores_df=scores_df, onsets=onsets, rankings=rankings)


def run_health_assessment(
    readings_df: Optional[pd.DataFrame],
    scores_df: Optional[pd.DataFrame] = None,
    onsets: Optional[Dict[str, Optional[int]]] = None,
) -> Tuple[Dict[str, SensorHealthReport], NetworkHealthSummary]:
    """Execute fleet-wide sensor operational health evaluation and network aggregation."""
    monitor = SensorHealthMonitor()
    reports = monitor.assess_all(readings_df, scores_df, onsets=onsets)
    summary = monitor.summarize_network_health(reports)
    return reports, summary


def run_maintenance_advisories(
    health_reports: Optional[Dict[str, Any]] = None,
    rankings: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None,
    investigation_report: Optional[Any] = None,
    scores_df: Optional[pd.DataFrame] = None,
    onsets: Optional[Dict[str, Optional[int]]] = None,
    suspect_sensor: Optional[str] = None,
    fault_type: Optional[str] = None,
    confidence_level: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, MaintenanceAdvisory], NetworkMaintenanceSummary]:
    """Generate predictive maintenance advisories across sensor fleet and aggregate network summary.

    Supports both keyword calls and legacy positional calls:
    (suspect_sensor, health_reports, fault_type, confidence_level).
    """
    advisor = MaintenanceAdvisor()

    # Support legacy positional call: run_maintenance_advisories(suspect_sensor, health_reports, ...)
    if isinstance(health_reports, str):
        suspect_s = health_reports
        reports = rankings if isinstance(rankings, dict) else kwargs.get("health_reports", {})
        f_type = investigation_report if isinstance(investigation_report, str) else fault_type
        c_level = scores_df if isinstance(scores_df, str) else confidence_level
        advisories = advisor.advise_fleet(
            suspect_sensor=suspect_s,
            health_reports=reports,
            fault_type=f_type,
            confidence_level=c_level,
        )
    else:
        advisories = advisor.advise_fleet(
            suspect_sensor=suspect_sensor,
            health_reports=health_reports or {},
            fault_type=fault_type,
            confidence_level=confidence_level,
            rankings=rankings,
            investigation_report=investigation_report,
            scores_df=scores_df,
            onsets=onsets,
            **kwargs,
        )

    summary = advisor.summarize_network_maintenance(advisories)
    return advisories, summary


# -----------------------------------------------------------------------------
# Global Live IoT Subsystem Singleton Manager
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Streamlit Secrets Ingestion & Live IoT Singleton
# -----------------------------------------------------------------------------

def load_mqtt_config_from_secrets(
    secrets_obj: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Safely load subscriber and publisher MQTT configuration from Streamlit secrets.

    Expected TOML structure in Streamlit Community Cloud:
    ```toml
    [mqtt]
    host = "kf64e014.ala.us-east-1.emqxsl.com"
    port = 8883
    topic = "faultmesh/sensors/#"
    keepalive = 60
    username = "faultmesh_subscriber"
    password = "SUBSCRIBER_PASSWORD"
    use_tls = true

    [mqtt.publisher]
    username = "faultmesh_user"
    password = "PUBLISHER_PASSWORD"
    ```

    Returns:
        Tuple of (sub_config: MQTTConnectionConfig, pub_kwargs: Dict[str, Any]).
    """
    import streamlit as st
    from faultmesh.ingestion.mqtt_client import MQTTConnectionConfig

    sub_config = MQTTConnectionConfig()
    pub_kwargs: Dict[str, Any] = {
        "broker_host": sub_config.host,
        "broker_port": sub_config.port,
        "use_tls": sub_config.use_tls,
        "ca_certs": sub_config.ca_certs,
        "username": None,
        "password": None,
    }

    try:
        # Resolve secrets mapping safely
        sec_root = secrets_obj if secrets_obj is not None else getattr(st, "secrets", None)
        if sec_root is not None and "mqtt" in sec_root:
            sec = sec_root["mqtt"]
            if hasattr(sec, "get") or isinstance(sec, dict):
                # Subscriber connection settings
                if "host" in sec and sec["host"]:
                    sub_config.host = str(sec["host"]).strip()
                if "port" in sec and sec["port"] is not None:
                    sub_config.port = int(sec["port"])
                if "topic" in sec and sec["topic"]:
                    sub_config.topic = str(sec["topic"]).strip()
                if "keepalive" in sec and sec["keepalive"] is not None:
                    sub_config.keepalive = int(sec["keepalive"])

                # TLS handling
                if "use_tls" in sec and sec["use_tls"] is not None:
                    sub_config.use_tls = bool(sec["use_tls"])
                elif sub_config.port == 8883:
                    sub_config.use_tls = True

                if "ca_certs" in sec and sec["ca_certs"]:
                    sub_config.ca_certs = str(sec["ca_certs"]).strip()

                # Subscriber authentication (never logged or exposed)
                if "username" in sec and sec["username"]:
                    sub_config.username = str(sec["username"]).strip()
                if "password" in sec and sec["password"] is not None:
                    sub_config.password = str(sec["password"])

                # Publisher authentication (separate [mqtt.publisher] section or fallback)
                pub_sec = sec.get("publisher", {})
                pub_user = None
                pub_pass = None
                if isinstance(pub_sec, (dict, object)) and hasattr(pub_sec, "get"):
                    if "username" in pub_sec and pub_sec["username"]:
                        pub_user = str(pub_sec["username"]).strip()
                    if "password" in pub_sec and pub_sec["password"] is not None:
                        pub_pass = str(pub_sec["password"])

                # If publisher credentials omitted, fallback to subscriber credentials
                if not pub_user and sub_config.username:
                    pub_user = sub_config.username
                if not pub_pass and sub_config.password:
                    pub_pass = sub_config.password

                pub_kwargs = {
                    "broker_host": sub_config.host,
                    "broker_port": sub_config.port,
                    "use_tls": sub_config.use_tls,
                    "ca_certs": sub_config.ca_certs,
                    "username": pub_user,
                    "password": pub_pass,
                }
    except Exception as exc:
        # Gracefully preserve local defaults if secrets are missing or malformed
        import logging
        logging.getLogger("faultmesh.dashboard").debug(f"Streamlit secrets unavailable or empty: {exc}")

    return sub_config, pub_kwargs


def get_global_live_iot_subsystem() -> Dict[str, Any]:
    """Provide a reliable process-wide singleton for Live IoT ingestion.

    Guarantees that Streamlit reruns, browser refreshes, and mode transitions
    always reference the exact same connected MQTT manager, Paho network loop,
    thread-safe message queue, and sensor buffer instances.
    """
    import streamlit as st
    from faultmesh.ingestion import (
        LiveTelemetryManager,
        DataValidator,
        MQTTClientManager,
        GatewaySimulator,
    )

    @st.cache_resource
    def _create_singleton() -> Dict[str, Any]:
        validator = DataValidator()
        telem_mgr = LiveTelemetryManager()
        sub_cfg, pub_kwargs = load_mqtt_config_from_secrets()
        mqtt_mgr = MQTTClientManager(
            config=sub_cfg,
            validator=validator,
            telemetry_manager=telem_mgr,
        )
        gw_sim = GatewaySimulator(
            **pub_kwargs,
        )
        return {
            "validator": validator,
            "telem_mgr": telem_mgr,
            "mqtt_mgr": mqtt_mgr,
            "gw_sim": gw_sim,
        }

    return _create_singleton()


