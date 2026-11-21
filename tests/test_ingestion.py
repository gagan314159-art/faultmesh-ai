"""Unit and integration tests for FaultMesh AI MQTT Ingestion architecture.

Tests:
    - Valid MQTT payload
    - Invalid JSON
    - Missing sensor ID
    - Unknown sensor
    - Invalid numeric reading
    - Incorrect measurement unit
    - Missing timestamp
    - Duplicate messages
    - Out-of-order messages
    - Stale readings
    - Data mapping
    - Rolling buffer
    - Calibration requirements
    - MQTT disconnection handling
    - Normal and faulty gateway simulation
"""

import json
import time
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from faultmesh.ingestion.data_validator import (
    DataValidator,
    SUPPORTED_SENSORS,
    STANDARD_UNITS,
)
from faultmesh.ingestion.data_mapper import DataMapper, TelemetryPoint
from faultmesh.ingestion.buffer import SensorBuffer, LiveTelemetryManager
from faultmesh.ingestion.mqtt_client import MQTTClientManager, MQTTConnectionConfig
from faultmesh.ingestion.gateway_simulator import GatewaySimulator


class TestDataValidator:
    """Test data validation against FaultMesh specifications."""

    def setup_method(self) -> None:
        self.validator = DataValidator()

    def test_valid_payload_all_sensors(self) -> None:
        """Verify valid payloads for all five supported sensors."""
        test_cases = [
            ("S1", 60.5, "°C"),
            ("S1", 61.0, "C"),
            ("S2", 2.15, "mm/s"),
            ("S3", 4.95, "bar"),
            ("S4", 30.2, "L/min"),
            ("S5", 10.1, "A"),
        ]
        now_iso = datetime.now(timezone.utc).isoformat()
        for sensor, val, unit in test_cases:
            payload = {
                "machine_id": "MACHINE_01",
                "sensor_id": sensor,
                "value": val,
                "unit": unit,
                "timestamp": now_iso,
            }
            res = self.validator.validate_message(payload)
            assert res.is_valid, f"Validation failed for {sensor}: {res.error}"
            assert res.validated_payload is not None
            assert res.validated_payload["sensor_id"] == sensor
            assert res.validated_payload["value"] == val
            assert res.validated_payload["unit"] == STANDARD_UNITS[sensor]

        assert self.validator.valid_count == len(test_cases)
        assert self.validator.rejected_count == 0

    def test_invalid_json(self) -> None:
        """Verify rejection of malformed JSON strings and byte payloads."""
        res1 = self.validator.validate_message("{bad_json: missing_quotes")
        assert not res1.is_valid
        assert res1.error_type == "MALFORMED_JSON"

        res2 = self.validator.validate_message(b"\xff\xfe\xfd not valid utf8")
        assert not res2.is_valid
        assert res2.error_type == "ENCODING_ERROR"

    def test_missing_sensor_id(self) -> None:
        """Verify rejection when sensor_id is missing."""
        payload = {
            "value": 60.0,
            "unit": "°C",
            "timestamp": "2026-11-21T16:30:00Z",
        }
        res = self.validator.validate_message(payload)
        assert not res.is_valid
        assert res.error_type == "MISSING_SENSOR_ID"

    def test_unknown_sensor(self) -> None:
        """Verify rejection of unsupported sensor identifiers."""
        payload = {
            "sensor_id": "S99",
            "value": 60.0,
            "unit": "°C",
            "timestamp": "2026-11-21T16:30:00Z",
        }
        res = self.validator.validate_message(payload)
        assert not res.is_valid
        assert res.error_type == "UNSUPPORTED_SENSOR"

    def test_invalid_numeric_reading(self) -> None:
        """Verify rejection of non-numeric, NaN, and Inf values."""
        invalid_values = ["not_a_number", None, float("nan"), float("inf"), True]
        for val in invalid_values:
            payload = {
                "sensor_id": "S1",
                "value": val,
                "unit": "°C",
                "timestamp": "2026-11-21T16:30:00Z",
            }
            res = self.validator.validate_message(payload)
            assert not res.is_valid
            assert res.error_type in ("INVALID_VALUE", "MISSING_VALUE")

    def test_incorrect_measurement_unit(self) -> None:
        """Verify rejection of wrong unit for sensor."""
        # S1 expects temperature (°C), not pressure (bar)
        payload = {
            "sensor_id": "S1",
            "value": 60.0,
            "unit": "bar",
            "timestamp": "2026-11-21T16:30:00Z",
        }
        res = self.validator.validate_message(payload)
        assert not res.is_valid
        assert res.error_type == "INVALID_UNIT"

    def test_missing_and_invalid_timestamp(self) -> None:
        """Verify rejection of corrupt timestamps while accepting valid string and numeric timestamps."""
        # Unparseable timestamp string must be rejected
        res = self.validator.validate_message({"sensor_id": "S1", "value": 60.0, "unit": "°C", "timestamp": "not-a-timestamp"})
        assert not res.is_valid
        assert res.error_type == "INVALID_TIMESTAMP"

    def test_python_dict_single_quotes_regression(self) -> None:
        """Regression test: stringified Python dictionaries with single quotes must parse safely."""
        raw = "{'machine_id': 'MACHINE_01', 'sensor_id': 'S1', 'value': 61.5, 'unit': 'C', 'timestamp': '2026-11-21T16:30:00Z'}"
        res = self.validator.validate_message(raw)
        assert res.is_valid, f"Failed on single-quote payload: {res.error}"
        assert res.validated_payload["sensor_id"] == "S1"
        assert res.validated_payload["value"] == 61.5
        assert res.validated_payload["unit"] == "°C"

    def test_powershell_unquoted_json_regression(self) -> None:
        """Regression test: payloads with stripped CLI/PowerShell quotes must be parsed safely."""
        raw = "{machine_id:MACHINE_01,sensor_id:S2,value:2.15,unit:mm/s,timestamp:2026-11-21T16:30:00Z}"
        res = self.validator.validate_message(raw)
        assert res.is_valid, f"Failed on unquoted CLI payload: {res.error}"
        assert res.validated_payload["sensor_id"] == "S2"
        assert res.validated_payload["value"] == 2.15

    def test_topic_sensor_inference_regression(self) -> None:
        """Regression test: when sensor_id is omitted from payload, infer from MQTT topic."""
        # Payload omits sensor_id, but topic is faultmesh/sensors/S3
        payload = '{"value": 5.12, "unit": "bar", "timestamp": "2026-11-21T16:30:00Z"}'
        res = self.validator.validate_message(payload, topic="faultmesh/sensors/S3")
        assert res.is_valid, f"Failed to infer sensor from topic: {res.error}"
        assert res.validated_payload["sensor_id"] == "S3"
        assert res.validated_payload["value"] == 5.12

        # Plain numeric payload on a sensor-specific topic
        res_num = self.validator.validate_message("30.45", topic="faultmesh/sensors/S4")
        assert res_num.is_valid, f"Failed on plain numeric payload: {res_num.error}"
        assert res_num.validated_payload["sensor_id"] == "S4"
        assert res_num.validated_payload["value"] == 30.45
        assert res_num.validated_payload["unit"] == "L/min"

    def test_field_aliases_regression(self) -> None:
        """Regression test: support common industrial field aliases (sensor, val, time)."""
        payload = {"sensor": "S5", "val": 10.4, "time": "2026-11-21T16:30:00Z"}
        res = self.validator.validate_message(payload)
        assert res.is_valid, f"Failed on field aliases: {res.error}"
        assert res.validated_payload["sensor_id"] == "S5"
        assert res.validated_payload["value"] == 10.4
        assert res.validated_payload["unit"] == "A"

    def test_millisecond_timestamp_regression(self) -> None:
        """Regression test: millisecond Unix timestamps (>1e11) must not trigger OSError on Windows."""
        payload = {"sensor_id": "S1", "value": 60.5, "unit": "°C", "timestamp": 1700000000000}
        res = self.validator.validate_message(payload)
        assert res.is_valid, f"Failed on ms timestamp: {res.error}"
        assert res.validated_payload["epoch"] == 1700000000.0

    def test_rejection_diagnostics_and_audit_history(self) -> None:
        """Verify that rejected payloads generate clear diagnostic logs and audit entries."""
        self.validator.reset_stats()
        # 1. Invalid unit
        r1 = self.validator.validate_message(
            {"sensor_id": "S1", "value": 60.0, "unit": "bar", "timestamp": "2026-11-21T16:30:00Z"},
            topic="faultmesh/sensors/S1",
        )
        assert not r1.is_valid
        assert r1.error_type == "INVALID_UNIT"

        # 2. Unknown sensor
        r2 = self.validator.validate_message(
            {"sensor_id": "S99", "value": 1.0, "timestamp": "2026-11-21T16:30:00Z"},
            topic="faultmesh/sensors/S99",
        )
        assert not r2.is_valid
        assert r2.error_type == "UNSUPPORTED_SENSOR"

        stats = self.validator.get_stats()
        assert stats["rejected_count"] == 2
        assert stats["error_rate_pct"] == 100.0
        assert len(stats["recent_rejections"]) == 2
        assert stats["recent_rejections"][0]["error_type"] == "INVALID_UNIT"
        assert stats["recent_rejections"][1]["error_type"] == "UNSUPPORTED_SENSOR"

    def test_gateway_simulator_payload_compatibility(self) -> None:
        """Verify that all five sensor payloads generated by GatewaySimulator pass validation."""
        gw = GatewaySimulator()
        now_iso = datetime.now(timezone.utc).isoformat()
        for sensor_id in SUPPORTED_SENSORS:
            payload = {
                "machine_id": "MACHINE_01",
                "sensor_id": sensor_id,
                "value": 50.0,
                "unit": STANDARD_UNITS[sensor_id],
                "timestamp": now_iso,
            }
            res = self.validator.validate_message(payload, topic=f"faultmesh/sensors/{sensor_id}")
            assert res.is_valid, f"Gateway payload failed for {sensor_id}: {res.error}"
            assert res.validated_payload["sensor_id"] == sensor_id



class TestDataMapper:
    """Test mapping validated payloads to TelemetryPoint and DataFrame."""

    def test_data_mapping_and_dataframe(self) -> None:
        validated = {
            "machine_id": "PLANT_A",
            "sensor_id": "S3",
            "value": 5.02,
            "unit": "bar",
            "timestamp": "2026-11-21T12:00:00Z",
            "epoch": 1795348800.0,
        }
        point = DataMapper.to_telemetry_point(validated)
        assert point.sensor_id == "S3"
        assert point.value == 5.02
        assert point.unit == "bar"

        # DataFrame mapping
        records = [
            {"time": 0, "S1": 60.1, "S2": 2.01, "S3": 5.0, "S4": 30.1, "S5": 10.0},
            {"time": 1, "S1": 60.2, "S2": 1.99, "S3": 5.02, "S4": 29.8, "S5": 10.05},
        ]
        df = DataMapper.records_to_dataframe(records)
        assert len(df) == 2
        assert list(df.columns[:6]) == ["time", "S1", "S2", "S3", "S4", "S5"]


class TestRollingBuffer:
    """Test sensor ring-buffer, deduplication, chronological ordering, and staleness."""

    def test_sensor_buffer_ordering_and_deduplication(self) -> None:
        buf = SensorBuffer("S1", max_size=10)
        base_epoch = 1000.0

        p1 = TelemetryPoint("M1", "S1", 60.0, "°C", "2026-11-21T00:00:01Z", base_epoch + 1.0)
        p2 = TelemetryPoint("M1", "S1", 60.2, "°C", "2026-11-21T00:00:02Z", base_epoch + 2.0)
        p_dup = TelemetryPoint("M1", "S1", 60.5, "°C", "2026-11-21T00:00:01Z", base_epoch + 1.0)
        p_ooo = TelemetryPoint("M1", "S1", 59.9, "°C", "2026-11-21T00:00:00Z", base_epoch + 0.0)

        assert buf.add_point(p1) is True
        assert buf.add_point(p2) is True
        # Duplicate should be rejected
        assert buf.add_point(p_dup) is False
        assert buf.duplicate_count == 1

        # Out-of-order should be placed at index 0
        assert buf.add_point(p_ooo) is True
        assert buf.out_of_order_count == 1

        points = buf.get_points()
        assert len(points) == 3
        assert points[0].value == 59.9  # Earliest epoch
        assert points[1].value == 60.0
        assert points[2].value == 60.2

    def test_stale_detection(self) -> None:
        buf = SensorBuffer("S2", max_size=10)
        assert buf.is_stale(threshold_seconds=5.0) is True

        p = TelemetryPoint("M1", "S2", 2.0, "mm/s", "2026-11-21T00:00:00Z", 1000.0)
        buf.add_point(p)
        assert buf.is_stale(threshold_seconds=5.0, current_time=time.time()) is False
        assert buf.is_stale(threshold_seconds=5.0, current_time=time.time() + 10.0) is True


class TestLiveTelemetryManagerAndCalibration:
    """Test live telemetry aggregation, baseline calibration workflow, and AI pipeline."""

    def test_calibration_lifecycle(self) -> None:
        mgr = LiveTelemetryManager(min_calibration_samples=25, target_calibration_samples=50)

        # Initially uncalibrated
        can_cal, count, msg = mgr.can_calibrate()
        assert can_cal is False
        assert count == 0
        assert mgr.operator_confirmed is False

        # Attempting analysis before calibration must not produce false diagnoses
        res = mgr.process_live_analysis()
        assert res["status"] == "awaiting_calibration"

        # Ingest 30 synchronized baseline samples across S1..S5
        base_epoch = 1700000000.0
        baselines = {"S1": 60.0, "S2": 2.0, "S3": 5.0, "S4": 30.0, "S5": 10.0}

        for i in range(35):
            t_iso = (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat()
            ep = base_epoch + i
            for s in SUPPORTED_SENSORS:
                # Add tiny Gaussian-like noise
                val = baselines[s] + (0.01 * (i % 3 - 1))
                pt = TelemetryPoint("M1", s, val, STANDARD_UNITS[s], t_iso, ep)
                mgr.ingest_point(pt)

        # Now can calibrate
        can_cal, count, msg = mgr.can_calibrate()
        assert can_cal is True
        assert count >= 35

        # Confirm and calibrate
        ok, cal_msg = mgr.confirm_and_calibrate()
        assert ok is True
        assert mgr.calibration_status == "calibrated"
        assert mgr.operator_confirmed is True
        assert mgr.calibrated_detector is not None

        # Process live analysis now succeeds
        analysis = mgr.process_live_analysis()
        assert analysis["status"] == "success"
        assert "rankings" in analysis
        assert "investigation_report" in analysis
        assert "health_reports" in analysis
        assert "advisories" in analysis

    def test_live_analysis_fault_localization(self) -> None:
        """Verify that after baseline calibration, an injected drift fault on S3 is correctly localized."""
        from faultmesh.simulator import SensorSimulator
        sim = SensorSimulator(seed=42)
        # Generate 120 steps with S3 drift fault starting at step 50
        df = sim.generate(n_steps=120, fault={"sensor": "S3", "type": "drift", "start": 50, "magnitude": 2.5})

        mgr = LiveTelemetryManager(min_calibration_samples=30, target_calibration_samples=45)
        base_epoch = 1700000000.0

        # Feed first 45 normal baseline samples
        for i in range(45):
            row = df.iloc[i]
            t_iso = (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat()
            ep = base_epoch + i
            for s in SUPPORTED_SENSORS:
                mgr.ingest_point(TelemetryPoint("M1", s, float(row[s]), STANDARD_UNITS[s], t_iso, ep))

        # Confirm and calibrate baseline on normal samples
        ok, _ = mgr.confirm_and_calibrate()
        assert ok is True

        # Ingest remaining samples (including S3 fault starting at t=50)
        for i in range(45, 120):
            row = df.iloc[i]
            t_iso = (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat()
            ep = base_epoch + i
            for s in SUPPORTED_SENSORS:
                mgr.ingest_point(TelemetryPoint("M1", s, float(row[s]), STANDARD_UNITS[s], t_iso, ep))

        # Execute live analysis
        analysis = mgr.process_live_analysis()
        assert analysis["status"] == "success"
        rankings = analysis["rankings"]
        assert len(rankings) > 0
        top_suspect = rankings[0][0]
        assert top_suspect == "S3"
        assert analysis["investigation_report"].suspected_sensor == "S3"



class TestMQTTClientAndSimulator:
    """Test MQTT client manager and software gateway simulator."""

    def test_mqtt_client_disconnection_handling(self) -> None:
        cfg = MQTTConnectionConfig(host="nonexistent.faultmesh.local", port=1883)
        client_mgr = MQTTClientManager(config=cfg)

        status_before = client_mgr.get_status()
        assert status_before["is_connected"] is False
        assert status_before["connection_state"] == "DISCONNECTED"

        # Disconnecting while disconnected should be graceful
        ok, msg = client_mgr.disconnect()
        assert ok is True

    def test_mqtt_queue_buffering_and_main_thread_drain(self) -> None:
        """Verify thread-safe queue buffering decouples background callback from main thread draining."""
        cfg = MQTTConnectionConfig(host="localhost", port=1883, topic="faultmesh/sensors/#")
        client_mgr = MQTTClientManager(config=cfg)

        # Mock Paho MQTT message object
        class MockMessage:
            def __init__(self, topic: str, payload: bytes):
                self.topic = topic
                self.payload = payload

        # Simulate background callback reception
        payload_s1 = b'{"machine_id":"MACHINE_01","sensor_id":"S1","value":61.5,"unit":"C","timestamp":"2026-11-21T16:30:00Z"}'
        mock_msg = MockMessage("faultmesh/sensors/S1", payload_s1)

        client_mgr._on_message_callback(None, None, mock_msg)

        # Before drain: callback recorded, message in queue, telemetry manager empty
        status = client_mgr.get_status()
        assert status["callbacks_received"] == 1
        assert status["pending_queue_size"] == 1
        assert status["messages_drained"] == 0
        assert status["last_received_topic"] == "faultmesh/sensors/S1"
        assert len(client_mgr.telemetry_manager.buffers["S1"].points) == 0

        # Drain on main thread
        drained = client_mgr.drain_message_queue(max_messages=10)
        assert drained == 1

        # After drain: queue empty, message drained counter incremented, telemetry buffer populated
        status_after = client_mgr.get_status()
        assert status_after["callbacks_received"] == 1
        assert status_after["pending_queue_size"] == 0
        assert status_after["messages_drained"] == 1
        assert len(client_mgr.telemetry_manager.buffers["S1"].points) == 1
        latest = client_mgr.telemetry_manager.get_latest_readings()
        assert latest["S1"]["value"] == 61.5

    def test_mqtt_topic_sensor_inference_via_queue(self) -> None:
        """Verify topic inference works seamlessly when payload sensor_id is omitted."""
        client_mgr = MQTTClientManager()

        class MockMessage:
            def __init__(self, topic: str, payload: bytes):
                self.topic = topic
                self.payload = payload

        # Payload without sensor_id, sensor specified in topic path
        payload = b'{"value": 4.95, "unit": "bar", "timestamp": "2026-11-21T16:30:00Z"}'
        mock_msg = MockMessage("faultmesh/sensors/S3", payload)

        client_mgr._on_message_callback(None, None, mock_msg)
        drained = client_mgr.drain_message_queue(max_messages=10)
        assert drained == 1

        assert len(client_mgr.telemetry_manager.buffers["S3"].points) == 1
        latest = client_mgr.telemetry_manager.get_latest_readings()
        assert latest["S3"]["value"] == 4.95

    def test_mqtt_on_subscribe_callback_and_status(self) -> None:
        """Verify on_subscribe callback records SUBACK status and timestamp."""
        client_mgr = MQTTClientManager()
        assert client_mgr.is_subscribed is False
        assert client_mgr.subscription_status == "Inactive"

        # Simulate Paho on_subscribe callback
        client_mgr._on_subscribe_callback(None, None, mid=1, granted_qos=[0])

        status = client_mgr.get_status()
        assert status["is_subscribed"] is True
        assert "Active" in status["subscription_status"]
        assert status["last_subscribed_at"] is not None

    def test_gateway_simulator_scenarios(self) -> None:
        """Verify Software Gateway Simulator sets scenarios correctly."""
        gw = GatewaySimulator(publish_interval=0.1)
        assert gw.is_running is False

        # Test scenario switching
        gw.set_scenario(scenario="normal")
        status = gw.get_status()
        assert status["active_scenario"] == "normal"

        gw.set_scenario(scenario="drift", fault_sensor="S3", magnitude_ratio=0.5)
        status = gw.get_status()
        assert status["active_scenario"] == "drift"
        assert status["fault_sensor"] == "S3"

        gw.set_scenario(scenario="spike", fault_sensor="S2")
        status = gw.get_status()
        assert status["active_scenario"] == "spike"

        gw.set_scenario(scenario="stuck", fault_sensor="S4")
        status = gw.get_status()
        assert status["active_scenario"] == "stuck"
