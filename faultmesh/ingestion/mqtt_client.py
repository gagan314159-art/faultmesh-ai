"""MQTT Client module for FaultMesh AI Live IoT Ingestion.

Responsibility:
    Connects to an MQTT broker, subscribes to configured telemetry topics,
    handles connection lifecycle, reconnects on transient outages,
    deposits incoming raw packets into a thread-safe message queue,
    and provides a drain mechanism for the Streamlit main thread.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import queue
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from faultmesh.ingestion.data_validator import DataValidator, ValidationResult
from faultmesh.ingestion.data_mapper import DataMapper
from faultmesh.ingestion.buffer import LiveTelemetryManager

logger = logging.getLogger("faultmesh.mqtt")

DEFAULT_BROKER_HOST: str = "localhost"
DEFAULT_BROKER_PORT: int = 1883
DEFAULT_BROKER_TOPIC: str = "faultmesh/sensors/#"
DEFAULT_KEEPALIVE: int = 60


@dataclass
class MQTTConnectionConfig:
    """Configuration container for MQTT broker parameters."""
    host: str = DEFAULT_BROKER_HOST
    port: int = DEFAULT_BROKER_PORT
    topic: str = DEFAULT_BROKER_TOPIC
    client_id: Optional[str] = None
    keepalive: int = DEFAULT_KEEPALIVE
    username: Optional[str] = None
    password: Optional[str] = None
    use_tls: bool = False
    ca_certs: Optional[str] = None


class MQTTClientManager:
    """Manages MQTT client lifecycle, thread-safe queue buffering, and callback dispatch."""

    def __init__(
        self,
        config: Optional[MQTTConnectionConfig] = None,
        validator: Optional[DataValidator] = None,
        telemetry_manager: Optional[LiveTelemetryManager] = None,
    ) -> None:
        """Initialize MQTT client manager with optional components."""
        self.config = config or MQTTConnectionConfig()
        self.validator = validator or DataValidator()
        self.telemetry_manager = telemetry_manager or LiveTelemetryManager()

        self.client: Optional[mqtt.Client] = None
        self.client_id: Optional[str] = None
        self.is_connected: bool = False
        self.is_subscribed: bool = False
        self.subscription_status: str = "Inactive"
        self.connection_state: str = "DISCONNECTED"  # DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR
        self.last_error: Optional[str] = None
        self.last_drain_error: Optional[str] = None
        self.last_connected_at: Optional[str] = None
        self.last_subscribed_at: Optional[str] = None
        self.last_message_at: Optional[str] = None
        self.last_received_topic: Optional[str] = None
        self.callbacks_received: int = 0
        self.messages_drained: int = 0

        # Thread-safe queue decoupling MQTT network daemon thread from Streamlit main thread
        self.message_queue: queue.Queue = queue.Queue(maxsize=2000)

        self.recent_logs: List[str] = []
        self._lock = threading.Lock()

    @property
    def messages_received(self) -> int:
        """Alias for callbacks_received for backward compatibility."""
        return self.callbacks_received

    def _log(self, message: str) -> None:
        """Append an event to the recent in-memory log buffer."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        with self._lock:
            self.recent_logs.append(entry)
            if len(self.recent_logs) > 60:
                self.recent_logs.pop(0)
        logger.info(message)

    def _create_paho_client(self, client_id: str) -> mqtt.Client:
        """Instantiate Paho client with version compatibility fallback."""
        try:
            # Check for Paho v2 API versioning
            from paho.mqtt.enums import CallbackAPIVersion
            client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=client_id)
            self._log(f"Initialized MQTT client '{client_id}' with CallbackAPIVersion.VERSION2")
        except Exception:
            # Fallback to legacy v1 API
            client = mqtt.Client(client_id=client_id)
            self._log(f"Initialized MQTT client '{client_id}' with legacy callback API")

        # Configure subscriber credentials if provided (never log passwords)
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
            self._log(f"Configured MQTT client authentication for user '{self.config.username}'")

        # Configure secure TLS encryption if enabled or default secure port 8883 is used
        if self.config.use_tls or self.config.port == 8883:
            import ssl
            try:
                if self.config.ca_certs:
                    client.tls_set(ca_certs=self.config.ca_certs)
                else:
                    client.tls_set()

                # Explicitly enforce certificate and hostname verification (never allow CERT_NONE)
                if hasattr(client, "_ssl_context") and client._ssl_context is not None:
                    client._ssl_context.check_hostname = True
                    client._ssl_context.verify_mode = ssl.CERT_REQUIRED

                ca_source = self.config.ca_certs or "Python default CA trust store"
                self._log(f"Configured secure TLS on client (CA: {ca_source}, verify: CERT_REQUIRED, hostname: True)")
            except Exception as tls_err:
                self._log(f"TLS configuration failure: {tls_err}")
                raise

        # Attach callbacks with flexible argument signatures
        client.on_connect = self._on_connect_callback
        client.on_subscribe = self._on_subscribe_callback
        client.on_disconnect = self._on_disconnect_callback
        client.on_message = self._on_message_callback
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        return client

    def _on_connect_callback(self, client: Any, userdata: Any, flags: Any, rc: Any, *args: Any, **kwargs: Any) -> None:
        """Handle MQTT connection established event with detailed lifecycle logging."""
        thread_name = threading.current_thread().name
        client_obj_id = hex(id(self))
        rc_code = getattr(rc, "value", rc)
        is_success = (rc_code == 0) or (hasattr(rc, "is_failure") and not rc.is_failure)

        if is_success:
            with self._lock:
                self.is_connected = True
                self.connection_state = "CONNECTED"
                self.last_connected_at = datetime.now(timezone.utc).isoformat()
            self._log(
                f"on_connect successful (rc={rc_code}) on thread '{thread_name}' | "
                f"Client: {client_obj_id} (ID: {self.client_id}) -> Broker {self.config.host}:{self.config.port}"
            )
            self._execute_subscription()
        else:
            with self._lock:
                self.is_connected = False
                self.is_subscribed = False
                self.connection_state = "ERROR"
                self.last_error = f"Connection refused (code {rc_code})"
            self._log(f"on_connect FAILED with return code {rc_code} on thread '{thread_name}' | Client: {client_obj_id}")

    def _execute_subscription(self) -> None:
        """Send MQTT subscribe packet with error checking."""
        if self.client is None or not self.is_connected:
            return
        target_topic = self.config.topic.strip()
        client_obj_id = hex(id(self))
        try:
            res, mid = self.client.subscribe(target_topic, qos=0)
            if res == mqtt.MQTT_ERR_SUCCESS:
                self._log(f"Subscribe packet sent for topic: '{target_topic}' (mid={mid}) | Client: {client_obj_id}")
            else:
                self._log(f"Subscribe failed with error code: {res} | Client: {client_obj_id}")
        except Exception as exc:
            self._log(f"Subscription exception on '{target_topic}': {exc} | Client: {client_obj_id}")

    def _on_subscribe_callback(self, client: Any = None, userdata: Any = None, mid: int = 0, *args: Any, **kwargs: Any) -> None:
        """Handle broker subscription acknowledgement (SUBACK). Supports Paho v1 and v2 callbacks."""
        thread_name = threading.current_thread().name
        client_obj_id = hex(id(self))
        with self._lock:
            self.is_subscribed = True
            self.subscription_status = f"Active ({self.config.topic})"
            self.last_subscribed_at = datetime.now(timezone.utc).isoformat()
        self._log(
            f"on_subscribe SUBACK confirmed (mid={mid}) on thread '{thread_name}' | "
            f"Topic: '{self.config.topic}' | Client: {client_obj_id}"
        )

    def _on_disconnect_callback(self, client: Any, userdata: Any, *args: Any, **kwargs: Any) -> None:
        """Handle MQTT disconnection event."""
        thread_name = threading.current_thread().name
        client_obj_id = hex(id(self))
        with self._lock:
            self.is_connected = False
            self.is_subscribed = False
            self.subscription_status = "Disconnected"
            if self.connection_state != "DISCONNECTED":
                self.connection_state = "RECONNECTING"
        self._log(f"Disconnected from MQTT broker on thread '{thread_name}' | Client: {client_obj_id} (reconnecting automatically...)")

    def _on_message_callback(self, client: Any, userdata: Any, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Handle incoming MQTT message in background thread: place in thread-safe queue."""
        thread_name = threading.current_thread().name
        client_obj_id = hex(id(self))
        queue_obj_id = hex(id(self.message_queue))
        topic = getattr(msg, "topic", "unknown")

        try:
            with self._lock:
                self.callbacks_received += 1
                self.last_message_at = datetime.now(timezone.utc).isoformat()
                self.last_received_topic = topic
                cb_count = self.callbacks_received

            item = {
                "topic": topic,
                "payload": getattr(msg, "payload", b""),
                "received_at": time.time(),
            }

            try:
                self.message_queue.put_nowait(item)
                insert_result = f"Enqueued (Pending queue size: {self.message_queue.qsize()})"
            except queue.Full:
                insert_result = "Queue Full (Dropping oldest item)"
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait(item)
                except Exception:
                    pass

            self._log(
                f"on_message callback #{cb_count} on thread '{thread_name}' for topic '{topic}' | "
                f"Client: {client_obj_id} | Queue: {queue_obj_id} | Status: {insert_result}"
            )

        except Exception as exc:
            self._log(f"Unexpected error in MQTT message callback: {exc}")

    def drain_message_queue(self, max_messages: int = 250) -> int:
        """Drain pending messages from thread-safe queue and process on Streamlit main thread.

        Args:
            max_messages: Maximum batch size to drain in a single call.

        Returns:
            Number of messages processed in this batch.
        """
        thread_name = threading.current_thread().name
        client_obj_id = hex(id(self))
        drained_count = 0
        while not self.message_queue.empty() and drained_count < max_messages:
            try:
                item = self.message_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as exc:
                self.last_drain_error = str(exc)
                break

            drained_count += 1
            with self._lock:
                self.messages_drained += 1

            raw_payload = item.get("payload", b"")
            topic = item.get("topic", "")

            # Validate payload
            try:
                val_res = self.validator.validate_message(raw_payload, topic=topic)

                if val_res.is_valid and val_res.validated_payload:
                    point = DataMapper.to_telemetry_point(val_res.validated_payload)
                    self.telemetry_manager.ingest_point(point)
                else:
                    raw_preview = val_res.raw_payload_preview or (str(raw_payload[:100]) if raw_payload else "<empty>")
                    self._log(f"Rejected telemetry on '{topic}' [{val_res.error_type}]: {val_res.error} | Raw: {raw_preview}")
            except Exception as exc:
                self.last_drain_error = f"Error processing item on '{topic}': {exc}"
                self._log(f"Drain processing error: {self.last_drain_error}")

        if drained_count > 0:
            self._log(
                f"Drained {drained_count} items on thread '{thread_name}' | "
                f"Client: {client_obj_id} | Remaining in Queue: {self.message_queue.qsize()}"
            )

        return drained_count

    def subscribe(self, new_topic: Optional[str] = None) -> Tuple[bool, str]:
        """Subscribe to a new topic filter while connected."""
        if new_topic:
            self.config.topic = new_topic.strip()

        with self._lock:
            if not self.is_connected or self.client is None:
                return False, "Cannot subscribe: MQTT client is not connected."

        self._execute_subscription()
        return True, f"Subscription requested for '{self.config.topic}'"

    def connect(self) -> Tuple[bool, str]:
        """Connect to the configured MQTT broker.

        Returns:
            Tuple of (success, message).
        """
        with self._lock:
            if self.is_connected:
                # Re-verify subscription if already connected
                self._execute_subscription()
                return True, f"Already connected to {self.config.host}:{self.config.port}"

            self.connection_state = "CONNECTING"
            self.last_error = None
            self.last_drain_error = None

        client_id = self.config.client_id or f"faultmesh_{uuid.uuid4().hex[:8]}"
        self.client_id = client_id
        client_obj_id = hex(id(self))
        queue_obj_id = hex(id(self.message_queue))

        try:
            self.client = self._create_paho_client(client_id)
            is_tls = bool(self.config.use_tls or self.config.port == 8883)
            self._log(
                f"Connecting to broker {self.config.host}:{self.config.port}... "
                f"[Client: {client_obj_id}, ID: {client_id}, Queue: {queue_obj_id}, TLS: {is_tls}]"
            )

            # Try direct connect first with short timeout, fallback to connect_async
            try:
                self.client.connect(
                    self.config.host,
                    port=self.config.port,
                    keepalive=self.config.keepalive,
                )
            except Exception:
                self.client.connect_async(
                    self.config.host,
                    port=self.config.port,
                    keepalive=self.config.keepalive,
                )

            self.client.loop_start()
            thread_info = getattr(getattr(self.client, "_thread", None), "name", "paho-mqtt-client")
            self._log(f"loop_start() active on thread '{thread_info}'.")
            return True, f"Connection established to {self.config.host}:{self.config.port}"
        except Exception as exc:
            with self._lock:
                self.is_connected = False
                self.connection_state = "ERROR"
                err_msg = str(exc)
                if "certificate verify failed" in err_msg.lower() or "ssl" in err_msg.lower() or "tls" in err_msg.lower():
                    self.last_error = f"TLS/SSL certificate or handshake error: {err_msg}"
                else:
                    self.last_error = err_msg
            self._log(f"Failed to connect to broker: {self.last_error}")
            return False, f"Could not connect to {self.config.host}:{self.config.port}: {self.last_error}"

    def disconnect(self) -> Tuple[bool, str]:
        """Cleanly disconnect from the MQTT broker."""
        with self._lock:
            self.connection_state = "DISCONNECTED"
            self.is_connected = False
            self.is_subscribed = False
            self.subscription_status = "Disconnected"

        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                self._log("Cleanly disconnected from MQTT broker.")
            except Exception as exc:
                self._log(f"Error during disconnection: {exc}")
            finally:
                self.client = None

        return True, "Disconnected successfully."

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive snapshot of MQTT connection, subscription, and queue telemetry."""
        with self._lock:
            logs = list(self.recent_logs)
            q_size = self.message_queue.qsize()

            # Paho network loop status inspection
            loop_alive = False
            loop_thread_name = "None"
            if self.client is not None and hasattr(self.client, "_thread") and self.client._thread is not None:
                loop_alive = self.client._thread.is_alive()
                loop_thread_name = self.client._thread.name

            if loop_alive:
                loop_status = f"Active ({loop_thread_name}, alive)"
            elif self.is_connected:
                loop_status = "Connected (network thread pending)"
            else:
                loop_status = "Stopped (disconnected)"

            return {
                "client_instance_id": f"MQTTClientManager@{hex(id(self))}",
                "client_id": self.client_id or (self.config.client_id or "Auto-generated"),
                "paho_client_ptr": hex(id(self.client)) if self.client else "None",
                "queue_identity": f"Queue@{hex(id(self.message_queue))}",
                "network_loop_status": loop_status,
                "is_network_loop_alive": loop_alive,
                "is_connected": self.is_connected,
                "connection_state": self.connection_state,
                "is_subscribed": self.is_subscribed,
                "subscription_status": self.subscription_status,
                "broker_host": self.config.host,
                "broker_port": self.config.port,
                "topic": self.config.topic,
                "use_tls": bool(self.config.use_tls or self.config.port == 8883),
                "has_auth": bool(self.config.username),
                "username": self.config.username or "None",
                "ca_certs": self.config.ca_certs or "System CA Store",
                "last_connected_at": self.last_connected_at,
                "last_subscribed_at": self.last_subscribed_at,
                "last_message_at": self.last_message_at,
                "last_received_topic": self.last_received_topic,
                "callbacks_received": self.callbacks_received,
                "messages_drained": self.messages_drained,
                "messages_received": self.callbacks_received,
                "pending_queue_size": q_size,
                "last_error": self.last_error,
                "last_drain_error": self.last_drain_error,
                "recent_logs": logs,
            }
