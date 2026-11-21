"""Software Gateway Simulator for FaultMesh AI Live IoT Demonstration.

Responsibility:
    Software-based MQTT publisher for local development, testing, and live
    demonstrations. Simulates multi-sensor industrial IoT gateway transmissions
    using the physics-grounded SensorSimulator.

    NOTICE:
    This is a Software Gateway Simulator for test and evaluation purposes.
    It does NOT connect to physical industrial hardware or PLCs.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import json
import logging
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from faultmesh.simulator import SensorSimulator, DEFAULT_BASELINES
from faultmesh.ingestion.data_validator import SUPPORTED_SENSORS, STANDARD_UNITS

logger = logging.getLogger("faultmesh.gateway_simulator")


class GatewaySimulator:
    """Software MQTT publisher simulating an edge IoT gateway for sensors S1-S5."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "faultmesh/sensors",
        machine_id: str = "MACHINE_01",
        publish_interval: float = 1.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        ca_certs: Optional[str] = None,
    ) -> None:
        """Initialize the gateway simulator."""
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix.rstrip("/")
        self.machine_id = machine_id
        self.publish_interval = max(0.1, publish_interval)
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.ca_certs = ca_certs

        self.is_running: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mqtt_client: Optional[mqtt.Client] = None

        self.step_index: int = 0
        self.messages_published: int = 0
        self.active_scenario: str = "normal"  # normal, drift, spike, stuck
        self.fault_sensor: str = "S3"
        self.fault_magnitude_ratio: float = 0.5
        self.fault_start_step: int = 20
        self.spike_duration: int = 10

        self._simulator: SensorSimulator = SensorSimulator(seed=int(time.time()) % 10000)
        self._precomputed_df: Optional[Any] = None
        self._lock = threading.Lock()

    def set_scenario(
        self,
        scenario: str = "normal",
        fault_sensor: str = "S3",
        magnitude_ratio: float = 0.5,
        fault_start_step: int = 20,
        spike_duration: int = 10,
    ) -> None:
        """Configure simulation scenario for live publishing."""
        with self._lock:
            self.active_scenario = scenario.lower()
            self.fault_sensor = fault_sensor.upper()
            self.fault_magnitude_ratio = float(magnitude_ratio)
            self.fault_start_step = int(fault_start_step)
            self.spike_duration = int(spike_duration)
            self._regenerate_stream()

    def _regenerate_stream(self) -> None:
        """Generate simulation telemetry stream using SensorSimulator."""
        base = DEFAULT_BASELINES.get(self.fault_sensor, 10.0)
        mag = base * self.fault_magnitude_ratio

        if self.active_scenario == "normal":
            fault_dict = None
        elif self.active_scenario == "drift":
            fault_dict = {
                "sensor": self.fault_sensor,
                "type": "drift",
                "start": self.fault_start_step,
                "magnitude": mag,
            }
        elif self.active_scenario == "spike":
            fault_dict = {
                "sensor": self.fault_sensor,
                "type": "spike",
                "start": self.fault_start_step,
                "magnitude": mag,
                "duration": self.spike_duration,
            }
        elif self.active_scenario == "stuck":
            fault_dict = {
                "sensor": self.fault_sensor,
                "type": "stuck",
                "start": self.fault_start_step,
                "magnitude": mag,
            }
        else:
            fault_dict = None

        # Re-initialize simulator with fresh seed
        self._simulator = SensorSimulator(seed=int(time.time() * 1000) % 65535)
        # Generate 600 steps of simulation data
        self._precomputed_df = self._simulator.generate(n_steps=600, fault=fault_dict)
        self.step_index = 0

    def start(self) -> Tuple[bool, str]:
        """Start the publisher background thread."""
        with self._lock:
            if self.is_running:
                return True, "Software Gateway Simulator is already running."

            self._regenerate_stream()
            self._stop_event.clear()
            self.is_running = True

        # Connect internal MQTT publishing client
        try:
            client_id = f"sim_gw_{uuid.uuid4().hex[:6]}"
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self._mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=client_id)
            except Exception:
                self._mqtt_client = mqtt.Client(client_id=client_id)

            # Independent publisher authentication
            if self.username:
                self._mqtt_client.username_pw_set(self.username, self.password)

            # Secure TLS configuration
            if self.use_tls or self.broker_port == 8883:
                import ssl
                if self.ca_certs:
                    self._mqtt_client.tls_set(ca_certs=self.ca_certs)
                else:
                    self._mqtt_client.tls_set()
                if hasattr(self._mqtt_client, "_ssl_context") and self._mqtt_client._ssl_context is not None:
                    self._mqtt_client._ssl_context.check_hostname = True
                    self._mqtt_client._ssl_context.verify_mode = ssl.CERT_REQUIRED

            self._mqtt_client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._mqtt_client.loop_start()
        except Exception as exc:
            with self._lock:
                self.is_running = False
            err_msg = str(exc)
            if "certificate verify failed" in err_msg.lower() or "ssl" in err_msg.lower() or "tls" in err_msg.lower():
                err_msg = f"TLS/SSL publisher error: {err_msg}"
            return False, f"Could not connect gateway publisher to MQTT broker: {err_msg}"

        self._thread = threading.Thread(target=self._publish_loop, daemon=True, name="GatewaySimulatorThread")
        self._thread.start()
        return True, f"Software Gateway Simulator started in '{self.active_scenario}' mode."

    def stop(self) -> Tuple[bool, str]:
        """Stop the background publisher cleanly."""
        with self._lock:
            if not self.is_running:
                return True, "Software Gateway Simulator is not running."
            self.is_running = False

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass
            self._mqtt_client = None

        return True, "Software Gateway Simulator stopped."

    def _publish_loop(self) -> None:
        """Main loop publishing multi-sensor readings periodically."""
        while not self._stop_event.is_set():
            loop_start = time.time()
            now_dt = datetime.now(timezone.utc)
            now_iso = now_dt.isoformat()

            with self._lock:
                if self._precomputed_df is None or self.step_index >= len(self._precomputed_df):
                    self._regenerate_stream()
                
                row = self._precomputed_df.iloc[self.step_index]
                self.step_index += 1

            # Publish S1..S5 readings
            for sensor_id in SUPPORTED_SENSORS:
                if self._stop_event.is_set():
                    break
                val = float(row[sensor_id])
                payload = {
                    "machine_id": self.machine_id,
                    "sensor_id": sensor_id,
                    "value": round(val, 4),
                    "unit": STANDARD_UNITS[sensor_id],
                    "timestamp": now_iso,
                }
                topic = f"{self.topic_prefix}/{sensor_id}"
                try:
                    if self._mqtt_client:
                        self._mqtt_client.publish(topic, json.dumps(payload), qos=0)
                        self.messages_published += 1
                except Exception as exc:
                    logger.warning(f"Gateway publishing error on {topic}: {exc}")

            # Sleep remaining interval
            elapsed = time.time() - loop_start
            sleep_time = max(0.01, self.publish_interval - elapsed)
            self._stop_event.wait(timeout=sleep_time)

    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary of the gateway simulator."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "step_index": self.step_index,
                "messages_published": self.messages_published,
                "active_scenario": self.active_scenario,
                "fault_sensor": self.fault_sensor,
                "publish_interval": self.publish_interval,
                "broker": f"{self.broker_host}:{self.broker_port}",
                "topic_prefix": self.topic_prefix,
                "use_tls": bool(self.use_tls or self.broker_port == 8883),
                "has_auth": bool(self.username),
                "username": self.username or "None",
            }
