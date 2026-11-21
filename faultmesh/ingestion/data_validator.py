"""Data validation module for FaultMesh AI MQTT Ingestion.

Responsibility:
    Strictly validate incoming MQTT JSON sensor telemetry messages.
    Ensures correct sensor identification, numeric fidelity, valid engineering units,
    and ISO-8601 timestamps. Tracks ingestion metrics, detailed error diagnostics,
    and maintains an audit log of rejected packets.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import re
import ast

# Supported sensor identifiers according to FaultMesh AI specifications
SUPPORTED_SENSORS: List[str] = ["S1", "S2", "S3", "S4", "S5"]

# Standard engineering units per sensor
STANDARD_UNITS: Dict[str, str] = {
    "S1": "°C",
    "S2": "mm/s",
    "S3": "bar",
    "S4": "L/min",
    "S5": "A",
}

# Permissible alias variations per sensor
PERMISSIBLE_UNITS: Dict[str, Set[str]] = {
    "S1": {"°C", "°c", "C", "c", "degC", "degc", "Celsius", "celsius"},
    "S2": {"mm/s", "MM/S", "mm/sec"},
    "S3": {"bar", "Bar", "BAR"},
    "S4": {"L/min", "l/min", "L/MIN", "lpm", "LPM"},
    "S5": {"A", "a", "Amps", "amps", "Amp", "amp"},
}


@dataclass
class ValidationResult:
    """Outcome of validating a single telemetry message."""
    is_valid: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    validated_payload: Optional[Dict[str, Any]] = None
    raw_payload_preview: Optional[str] = None

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Convenience alias for validated payload."""
        return self.validated_payload


class DataValidator:
    """Validates raw MQTT telemetry payloads and maintains validation telemetry counters."""

    def __init__(self, max_rejection_history: int = 25) -> None:
        """Initialize validator and performance counters."""
        self.total_received: int = 0
        self.valid_count: int = 0
        self.rejected_count: int = 0
        self.errors_by_type: Dict[str, int] = {}
        self.counts_by_sensor: Dict[str, int] = {s: 0 for s in SUPPORTED_SENSORS}
        self.last_error: Optional[str] = None
        self.last_rejected_payload: Optional[str] = None
        self.last_rejected_topic: Optional[str] = None
        self.last_valid_time: Optional[datetime] = None
        self.recent_rejections: List[Dict[str, Any]] = []
        self.max_rejection_history = max_rejection_history

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.total_received = 0
        self.valid_count = 0
        self.rejected_count = 0
        self.errors_by_type.clear()
        self.counts_by_sensor = {s: 0 for s in SUPPORTED_SENSORS}
        self.last_error = None
        self.last_rejected_payload = None
        self.last_rejected_topic = None
        self.last_valid_time = None
        self.recent_rejections.clear()

    def _record_error(
        self,
        error_type: str,
        message: str,
        raw_preview: str = "",
        topic: Optional[str] = None,
    ) -> ValidationResult:
        """Record a validation failure and return failure result."""
        self.rejected_count += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        self.last_error = f"[{error_type}] {message}"
        self.last_rejected_payload = raw_preview
        self.last_rejected_topic = topic

        rejection_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat()[:19],
            "topic": topic or "N/A",
            "error_type": error_type,
            "reason": message,
            "payload_snippet": raw_preview[:120] if raw_preview else "<empty>",
        }
        self.recent_rejections.append(rejection_entry)
        if len(self.recent_rejections) > self.max_rejection_history:
            self.recent_rejections.pop(0)

        return ValidationResult(
            is_valid=False,
            error=message,
            error_type=error_type,
            validated_payload=None,
            raw_payload_preview=raw_preview,
        )

    @staticmethod
    def _extract_sensor_from_topic(topic: Optional[str]) -> Optional[str]:
        """Infer sensor ID from MQTT topic path (e.g. faultmesh/sensors/S1 -> S1)."""
        if not topic:
            return None
        parts = topic.strip("/").split("/")
        for part in reversed(parts):
            u = part.upper()
            if u in SUPPORTED_SENSORS:
                return u
        return None

    @staticmethod
    def _parse_unquoted_kv(text: str) -> Optional[Dict[str, Any]]:
        """Parse key-value pair string resulting from CLI or PowerShell quote stripping."""
        clean = text.strip()
        if not (clean.startswith("{") and clean.endswith("}")):
            return None
        inner = clean[1:-1].strip()
        # Split by comma where next token looks like a key
        parts = re.split(r',\s*(?=[a-zA-Z0-9_]+\s*:)', inner)
        d: Dict[str, Any] = {}
        for part in parts:
            if ":" not in part:
                return None
            k, v = part.split(":", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip().strip('"').strip("'")
            # Try numeric parse
            try:
                if "." in v or "e" in v.lower():
                    d[k] = float(v)
                else:
                    d[k] = int(v)
            except ValueError:
                d[k] = v
        return d if d else None

    def validate_message(
        self,
        raw_payload: Union[str, bytes, dict, Any],
        topic: Optional[str] = None,
    ) -> ValidationResult:
        """Validate an incoming MQTT payload against FaultMesh specifications.

        Args:
            raw_payload: Raw string, bytes, or parsed dict from MQTT topic.
            topic: Optional MQTT topic string to extract sensor ID fallback.

        Returns:
            ValidationResult with sanitized data or rejection reason.
        """
        self.total_received += 1

        # 1. Parse JSON if string or bytes
        raw_preview = ""
        payload: Optional[Dict[str, Any]] = None

        if isinstance(raw_payload, (bytes, bytearray)):
            try:
                decoded = raw_payload.decode("utf-8")
                raw_preview = decoded[:150]
            except UnicodeDecodeError as exc:
                return self._record_error("ENCODING_ERROR", f"Payload is not valid UTF-8: {exc}", topic=topic)
            try:
                payload = json.loads(decoded)
            except json.JSONDecodeError as exc:
                # Attempt fallback for Python single-quoted dict or PowerShell unquoted JSON
                if decoded.strip().startswith("{") and decoded.strip().endswith("}"):
                    try:
                        parsed = ast.literal_eval(decoded.strip())
                        if isinstance(parsed, dict):
                            payload = parsed
                    except Exception:
                        pass
                if payload is None:
                    parsed_kv = self._parse_unquoted_kv(decoded)
                    if parsed_kv:
                        payload = parsed_kv
                if payload is None:
                    return self._record_error("MALFORMED_JSON", f"Malformed JSON: {exc.msg} | Received: '{raw_preview[:60]}'", raw_preview, topic=topic)

        elif isinstance(raw_payload, str):
            clean_str = raw_payload.strip()
            raw_preview = clean_str[:150]
            try:
                payload = json.loads(clean_str)
            except json.JSONDecodeError as exc:
                # Attempt fallback for Python single-quoted dict or PowerShell unquoted JSON
                if clean_str.startswith("{") and clean_str.endswith("}"):
                    try:
                        parsed = ast.literal_eval(clean_str)
                        if isinstance(parsed, dict):
                            payload = parsed
                    except Exception:
                        pass
                if payload is None:
                    parsed_kv = self._parse_unquoted_kv(clean_str)
                    if parsed_kv:
                        payload = parsed_kv
                # Check if plain numeric value on a sensor-specific topic (e.g. topic 'faultmesh/sensors/S1' with body '61.5')
                if payload is None:
                    try:
                        val = float(clean_str)
                        sensor_from_topic = self._extract_sensor_from_topic(topic)
                        if sensor_from_topic and not (math.isnan(val) or math.isinf(val)):
                            payload = {"sensor_id": sensor_from_topic, "value": val}
                    except ValueError:
                        pass
                if payload is None:
                    return self._record_error("MALFORMED_JSON", f"Malformed JSON: {exc.msg} | Received: '{raw_preview[:60]}'", raw_preview, topic=topic)

        elif isinstance(raw_payload, dict):
            payload = raw_payload
            raw_preview = str(payload)[:150]
        else:
            return self._record_error(
                "INVALID_TYPE",
                f"Payload must be string, bytes, or dict, got {type(raw_payload).__name__}",
                raw_preview=str(raw_payload)[:100],
                topic=topic,
            )

        if not isinstance(payload, dict):
            # Check if payload is bare numeric and topic contains sensor ID
            if isinstance(payload, (int, float)):
                sensor_from_topic = self._extract_sensor_from_topic(topic)
                if sensor_from_topic:
                    payload = {"sensor_id": sensor_from_topic, "value": float(payload)}
                else:
                    return self._record_error("INVALID_FORMAT", "Payload root must be a JSON object", raw_preview, topic=topic)
            else:
                return self._record_error("INVALID_FORMAT", f"Payload root must be a JSON object, got {type(payload).__name__}", raw_preview, topic=topic)

        # 2. Extract and check sensor_id
        raw_sensor_id = payload.get("sensor_id") or payload.get("sensor") or payload.get("id")
        topic_sensor = self._extract_sensor_from_topic(topic)

        # Reject conflicting sensor IDs between payload and topic path
        if raw_sensor_id and topic_sensor:
            candidate_id = str(raw_sensor_id).strip().upper()
            if candidate_id in SUPPORTED_SENSORS and candidate_id != topic_sensor:
                return self._record_error(
                    "CONFLICTING_SENSOR_ID",
                    f"Payload sensor ID '{raw_sensor_id}' conflicts with MQTT topic sensor '{topic_sensor}'",
                    raw_preview,
                    topic=topic,
                )

        if not raw_sensor_id and topic:
            raw_sensor_id = topic_sensor

        if not raw_sensor_id:
            return self._record_error("MISSING_SENSOR_ID", "Missing required field 'sensor_id' (and not inferrable from topic)", raw_preview, topic=topic)

        sensor_id = str(raw_sensor_id).strip().upper()
        if sensor_id not in SUPPORTED_SENSORS:
            return self._record_error(
                "UNSUPPORTED_SENSOR",
                f"Unsupported sensor ID '{raw_sensor_id}'. Must be one of {SUPPORTED_SENSORS}",
                raw_preview,
                topic=topic,
            )

        # 3. Check numeric value
        raw_val = payload.get("value")
        if raw_val is None:
            # Check aliases
            raw_val = payload.get("val", payload.get("reading"))

        if raw_val is None:
            return self._record_error("MISSING_VALUE", f"Missing required numeric field 'value' for sensor {sensor_id}", raw_preview, topic=topic)

        if isinstance(raw_val, bool):
            return self._record_error("INVALID_VALUE", "Sensor value cannot be a boolean", raw_preview, topic=topic)

        try:
            val_float = float(raw_val)
        except (ValueError, TypeError):
            return self._record_error("INVALID_VALUE", f"Value '{raw_val}' cannot be converted to float", raw_preview, topic=topic)

        if math.isnan(val_float) or math.isinf(val_float):
            return self._record_error("INVALID_VALUE", f"Value '{raw_val}' is NaN or infinite", raw_preview, topic=topic)

        # 4. Check measurement unit
        standard_unit = STANDARD_UNITS[sensor_id]
        raw_unit_str: Optional[str] = payload.get("unit")

        if raw_unit_str is not None and str(raw_unit_str).strip() != "":
            raw_unit = str(raw_unit_str).strip()
            allowed_units = PERMISSIBLE_UNITS.get(sensor_id, set())
            if raw_unit not in allowed_units:
                return self._record_error(
                    "INVALID_UNIT",
                    f"Invalid unit '{raw_unit}' for sensor {sensor_id}. Expected one of: {sorted(list(allowed_units))}",
                    raw_preview,
                    topic=topic,
                )
        else:
            # Unit omitted: safe default to canonical unit defined for this sensor
            raw_unit = standard_unit

        # 5. Check timestamp
        raw_ts = payload.get("timestamp") or payload.get("time") or payload.get("ts")
        parsed_dt: Optional[datetime] = None
        epoch_sec: Optional[float] = None

        if raw_ts is None:
            # Default to packet ingestion arrival time
            parsed_dt = datetime.now(timezone.utc)
            epoch_sec = parsed_dt.timestamp()
        elif isinstance(raw_ts, (int, float)):
            try:
                epoch_val = float(raw_ts)
                # Handle millisecond timestamps (e.g. 1.7e12)
                if epoch_val > 1e11:
                    epoch_val = epoch_val / 1000.0
                parsed_dt = datetime.fromtimestamp(epoch_val, tz=timezone.utc)
                epoch_sec = epoch_val
            except Exception as exc:
                return self._record_error("INVALID_TIMESTAMP", f"Invalid numeric epoch timestamp: {exc}", raw_preview, topic=topic)
        elif isinstance(raw_ts, str):
            clean_ts = raw_ts.strip()
            # Replace trailing 'Z' with '+00:00' for standard fromisoformat
            iso_str = clean_ts.replace("Z", "+00:00") if clean_ts.endswith("Z") else clean_ts
            try:
                parsed_dt = datetime.fromisoformat(iso_str)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                epoch_sec = parsed_dt.timestamp()
            except Exception:
                # Fallback string formats
                formats = [
                    "%Y-%m-%dT%H:%M:%S.%f%z",
                    "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                ]
                for fmt in formats:
                    try:
                        parsed_dt = datetime.strptime(clean_ts, fmt).replace(tzinfo=timezone.utc)
                        epoch_sec = parsed_dt.timestamp()
                        break
                    except ValueError:
                        pass

                if parsed_dt is None or epoch_sec is None:
                    return self._record_error("INVALID_TIMESTAMP", f"Unrecognized timestamp string: '{raw_ts}'", raw_preview, topic=topic)
        else:
            return self._record_error("INVALID_TIMESTAMP", f"Timestamp must be ISO string or numeric epoch, got {type(raw_ts).__name__}", raw_preview, topic=topic)

        # Machine ID (optional default)
        machine_id = str(payload.get("machine_id", "MACHINE_01")).strip()

        # Success: build sanitized dictionary
        sanitized = {
            "machine_id": machine_id,
            "sensor_id": sensor_id,
            "value": round(val_float, 4),
            "unit": standard_unit,
            "raw_unit": raw_unit,
            "timestamp": parsed_dt.isoformat(),
            "epoch": epoch_sec,
        }

        self.valid_count += 1
        self.counts_by_sensor[sensor_id] = self.counts_by_sensor.get(sensor_id, 0) + 1
        self.last_valid_time = datetime.now(timezone.utc)

        return ValidationResult(
            is_valid=True,
            error=None,
            error_type=None,
            validated_payload=sanitized,
            raw_payload_preview=raw_preview,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return snapshot dictionary of validation telemetry and rejection audit logs."""
        return {
            "total_received": self.total_received,
            "valid_count": self.valid_count,
            "rejected_count": self.rejected_count,
            "error_rate_pct": round((self.rejected_count / self.total_received * 100.0) if self.total_received > 0 else 0.0, 2),
            "errors_by_type": dict(self.errors_by_type),
            "counts_by_sensor": dict(self.counts_by_sensor),
            "last_error": self.last_error,
            "last_rejected_payload": self.last_rejected_payload,
            "last_rejected_topic": self.last_rejected_topic,
            "last_valid_time": self.last_valid_time.isoformat() if self.last_valid_time else None,
            "recent_rejections": list(self.recent_rejections),
        }
