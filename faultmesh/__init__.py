"""FaultMesh AI: Intelligent Cooperative IoT Fault Localization System.

This package provides modules for:
- Simulating multi-sensor IoT time series with noise, fault injection, and neighbor coupling.
- Feature extraction with rolling statistics and normalization.
- Anomaly detection using Isolation Forest models.
- Graph-based sensor network topology modeling.
- Multi-criteria fault localization and root-cause ranking.
- End-to-end simulation, detection, and localization pipeline orchestration.
- SQLite persistence for sensor readings, alerts, and evaluation reports.
"""

__version__ = "2.0.0"
__author__ = "FaultMesh AI Team"

from faultmesh.simulator import SensorSimulator
from faultmesh.processing import FeatureProcessor
from faultmesh.detector import AnomalyDetector
from faultmesh.graph import SensorNetworkGraph
from faultmesh.localizer import FaultLocalizer
from faultmesh.pipeline import FaultMeshPipeline
from faultmesh.storage import StorageManager
from faultmesh.explainer import FaultExplainer
from faultmesh.health import SensorHealthMonitor, SensorHealthRecord, NetworkHealthSummary
from faultmesh.maintenance import MaintenanceAdvisor
from faultmesh.reporting import (
    ExecutiveReporter,
    build_investigation_summary,
    export_summary_csv,
    generate_pdf_report,
)
from faultmesh.ingestion import (
    DataValidator,
    DataMapper,
    TelemetryPoint,
    LiveTelemetryManager,
    MQTTClientManager,
    MQTTConnectionConfig,
    GatewaySimulator,
)

__all__ = [
    "SensorSimulator",
    "FeatureProcessor",
    "AnomalyDetector",
    "SensorNetworkGraph",
    "FaultLocalizer",
    "FaultMeshPipeline",
    "StorageManager",
    "FaultExplainer",
    "SensorHealthMonitor",
    "SensorHealthRecord",
    "NetworkHealthSummary",
    "MaintenanceAdvisor",
    "ExecutiveReporter",
    "build_investigation_summary",
    "export_summary_csv",
    "generate_pdf_report",
    "DataValidator",
    "DataMapper",
    "TelemetryPoint",
    "LiveTelemetryManager",
    "MQTTClientManager",
    "MQTTConnectionConfig",
    "GatewaySimulator",
]
