import time
import subprocess
import json
from datetime import datetime, timezone
from streamlit.testing.v1 import AppTest

print("--- Step 1: Initializing AppTest on dashboard/app.py ---")
at = AppTest.from_file("dashboard/app.py", default_timeout=30)
at.run()
assert not at.exception, f"App exception: {at.exception}"
print("Dashboard initialized successfully.")

print("--- Step 2: Selecting Live IoT Mode ---")
# System Operating Mode is radio[0]
mode_radio = at.sidebar.radio[0]
print(f"Current mode: {mode_radio.value}, options: {mode_radio.options}")
mode_radio.set_value("Live IoT Mode").run()
assert not at.exception, f"Exception after mode switch: {at.exception}"
print(f"Switched mode to: {at.sidebar.radio[0].value}")

print("--- Step 3: Clicking Connect to Mosquitto on localhost:1883 ---")
# Find Connect button in sidebar
connect_btn = None
for btn in at.sidebar.button:
    if btn.label == "Connect":
        connect_btn = btn
        break

assert connect_btn is not None, "Connect button not found in sidebar"
connect_btn.click().run()
assert not at.exception, f"Exception after clicking Connect: {at.exception}"

# Wait 1 second for TCP handshake and SUBACK
time.sleep(1)
at.run()

# Check MQTT manager status from session state
mqtt_mgr = at.session_state.mqtt_mgr
status = mqtt_mgr.get_status()
print("MQTT Manager Status after Connect:")
for k, v in status.items():
    if k != "recent_logs":
        print(f"  {k}: {v}")

assert status["is_connected"] is True, f"Expected connected, got: {status['connection_state']}"
assert status["is_subscribed"] is True, f"Expected subscribed, got: {status['subscription_status']}"
assert status["topic"] == "faultmesh/sensors/#"

print("--- Step 4: Publishing S1 Telemetry via mosquitto_pub.exe ---")
now_iso = datetime.now(timezone.utc).isoformat()
payload_s1 = json.dumps({
    "machine_id": "MACHINE_01",
    "sensor_id": "S1",
    "value": 61.5,
    "unit": "C",
    "timestamp": now_iso
})

pub1 = subprocess.run(
    [r"C:\Program Files\mosquitto\mosquitto_pub.exe", "-h", "localhost", "-p", "1883", "-t", "faultmesh/sensors/S1", "-m", payload_s1],
    capture_output=True,
    text=True
)
print("mosquitto_pub S1 returncode:", pub1.returncode, pub1.stderr)
assert pub1.returncode == 0, f"mosquitto_pub failed: {pub1.stderr}"

# Wait 0.5s for Paho callback to receive and enqueue
time.sleep(0.5)

# Rerun Streamlit app to drain queue and update UI
at.run()
assert not at.exception, f"Exception during rerun: {at.exception}"

status_after_s1 = mqtt_mgr.get_status()
stats_s1 = mqtt_mgr.validator.get_stats()
print("\n=== STATUS AFTER S1 PUBLICATION ===")
print("  Callbacks Received:", status_after_s1["callbacks_received"])
print("  Messages Drained:", status_after_s1["messages_drained"])
print("  Valid Count:", stats_s1["valid_count"])
print("  Rejected Count:", stats_s1["rejected_count"])
print("  Pending Queue Size:", status_after_s1["pending_queue_size"])
print("  Last Received Topic:", status_after_s1["last_received_topic"])
print("  Last Message Time:", status_after_s1["last_message_at"])

assert status_after_s1["callbacks_received"] >= 1, "Callbacks received must be >= 1"
assert stats_s1["valid_count"] >= 1, "Valid count must be >= 1"

print("\n--- Step 5: Publishing S2, S3, S4, and S5 Telemetry via mosquitto_pub.exe ---")
remaining_sensors = [
    ("S2", 2.15, "mm/s"),
    ("S3", 4.95, "bar"),
    ("S4", 30.2, "L/min"),
    ("S5", 10.1, "A"),
]

for s_id, s_val, s_unit in remaining_sensors:
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = json.dumps({
        "machine_id": "MACHINE_01",
        "sensor_id": s_id,
        "value": s_val,
        "unit": s_unit,
        "timestamp": now_iso
    })
    pub = subprocess.run(
        [r"C:\Program Files\mosquitto\mosquitto_pub.exe", "-h", "localhost", "-p", "1883", "-t", f"faultmesh/sensors/{s_id}", "-m", payload],
        capture_output=True,
        text=True
    )
    assert pub.returncode == 0, f"mosquitto_pub failed for {s_id}: {pub.stderr}"

# Wait 0.5s for Paho to receive all packets
time.sleep(0.5)

# Rerun app
at.run()
assert not at.exception, f"Exception during rerun: {at.exception}"

final_status = mqtt_mgr.get_status()
final_stats = mqtt_mgr.validator.get_stats()
latest_readings = at.session_state.telem_mgr.get_latest_readings()

print("\n=== FINAL DASHBOARD & SENSOR READINGS STATE ===")
print("  Total Callbacks Received:", final_status["callbacks_received"])
print("  Total Messages Drained:", final_status["messages_drained"])
print("  Valid Sensor Readings:", final_stats["valid_count"])
print("  Rejected Packets:", final_stats["rejected_count"])
print("  Latest Sensor Observations:")
for s_id, r in latest_readings.items():
    print(f"    {s_id}: value={r.get('value')} {r.get('unit')}, updated_at={r.get('timestamp')}")

# Verify all 5 sensors S1..S5 are present in the buffer
for s_id in ["S1", "S2", "S3", "S4", "S5"]:
    assert s_id in latest_readings, f"Sensor {s_id} missing from latest readings!"
    assert latest_readings[s_id]["value"] is not None, f"Sensor {s_id} reading is None!"

print("\nRecent Logs from MQTT Manager:")
for l in final_status["recent_logs"][-10:]:
    print("  ", l)

print("\nALL 5 SENSORS (S1, S2, S3, S4, S5) SUCCESSFULLY INGESTED AND VERIFIED!")

# Clean disconnect
mqtt_mgr.disconnect()
print("Disconnected cleanly.")
