"""Data mapping module for FaultMesh AI MQTT Ingestion.

Responsibility:
    Converts validated MQTT sensor telemetry records into the standard
    data structures and DataFrames required by the FaultMesh AI diagnostic
    and analytical pipelines without altering sensor semantics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from faultmesh.ingestion.data_validator import SUPPORTED_SENSORS, STANDARD_UNITS

# Canonical sensor metadata mapping
SENSOR_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "S1": {"name": "Bearing Temperature", "unit": "°C", "system": "Bearing Housing", "baseline": "60.0"},
    "S2": {"name": "Shaft Vibration", "unit": "mm/s", "system": "Drive Shaft", "baseline": "2.0"},
    "S3": {"name": "Cooling Pressure", "unit": "bar", "system": "Cooling Manifold", "baseline": "5.0"},
    "S4": {"name": "Coolant Flow", "unit": "L/min", "system": "Coolant Delivery", "baseline": "30.0"},
    "S5": {"name": "Motor Current", "unit": "A", "system": "Drive Motor Phase", "baseline": "10.0"},
}


@dataclass
class TelemetryPoint:
    """Standardized single-sensor telemetry observation."""
    machine_id: str
    sensor_id: str
    value: float
    unit: str
    timestamp: str
    epoch: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "sensor_id": self.sensor_id,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "epoch": self.epoch,
        }


class DataMapper:
    """Transforms validated telemetry payloads into pipeline-compatible formats."""

    @staticmethod
    def to_telemetry_point(validated_dict: Dict[str, Any]) -> TelemetryPoint:
        """Create TelemetryPoint from validated dictionary."""
        return TelemetryPoint(
            machine_id=str(validated_dict.get("machine_id", "MACHINE_01")),
            sensor_id=str(validated_dict["sensor_id"]),
            value=float(validated_dict["value"]),
            unit=str(validated_dict.get("unit", STANDARD_UNITS.get(validated_dict["sensor_id"], ""))),
            timestamp=str(validated_dict["timestamp"]),
            epoch=float(validated_dict["epoch"]),
        )

    @staticmethod
    def get_sensor_info(sensor_id: str) -> Dict[str, str]:
        """Return engineering definition of the specified sensor."""
        clean_id = sensor_id.upper()
        if clean_id in SENSOR_DEFINITIONS:
            return dict(SENSOR_DEFINITIONS[clean_id])
        return {"name": f"Sensor {sensor_id}", "unit": "units", "system": "Generic", "baseline": "0.0"}

    @staticmethod
    def records_to_dataframe(
        aligned_records: List[Dict[str, Any]],
        sensor_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert a list of synchronized multi-sensor observations into a pipeline DataFrame.

        Expected record format:
        {'time': 0, 'S1': 60.1, 'S2': 2.01, 'S3': 4.98, 'S4': 30.2, 'S5': 10.05, 'timestamp': '...'}

        Returns:
            DataFrame with 'time' column and sensor columns S1..S5.
        """
        sensors = sensor_cols or SUPPORTED_SENSORS
        if not aligned_records:
            cols = ["time"] + sensors
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(aligned_records)
        if "time" not in df.columns:
            df["time"] = np.arange(len(df), dtype=int)

        # Ensure all expected sensor columns exist and are numeric
        for s in sensors:
            if s not in df.columns:
                df[s] = np.nan
            else:
                df[s] = pd.to_numeric(df[s], errors="coerce")

        cols = ["time"] + [s for s in sensors if s in df.columns]
        extra_cols = [c for c in df.columns if c not in cols]
        return df[cols + extra_cols]
