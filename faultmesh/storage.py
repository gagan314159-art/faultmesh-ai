"""SQLite storage module for FaultMesh AI.

Responsibility:
    Store raw sensor time-series readings, anomaly detection alerts, and
    fault localization reports in SQLite database using Python's built-in sqlite3 module.
"""

from typing import Dict, List, Optional, Any
import os
import json
import sqlite3
import pandas as pd


DEFAULT_DB_PATH = "data/faultmesh.db"


def _ensure_dir(db_path: str) -> None:
    """Ensure the directory for the SQLite database exists."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database with required tables if they do not exist.

    Tables created:
        1. readings: Stores time-series observations (time, sensor, value).
        2. alerts: Records detection triggers (id, sensor, time, anomaly_score, message).
        3. fault_reports: Saves localization diagnostic outputs (id, created_at, report_json).

    Args:
        db_path: Path to SQLite database file (default: 'data/faultmesh.db').
    """
    _ensure_dir(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS readings (
                    time INTEGER,
                    sensor TEXT,
                    value REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor TEXT,
                    time INTEGER,
                    anomaly_score REAL,
                    message TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fault_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    report_json TEXT
                )
            """)
            # Compatibility tables for legacy test scenarios
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestep INTEGER,
                    sensor_id TEXT,
                    reading_value REAL,
                    is_faulty BOOLEAN
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestep INTEGER,
                    sensor_id TEXT,
                    anomaly_score REAL,
                    alarm_triggered BOOLEAN
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ground_truth_sensor TEXT,
                    fault_type TEXT,
                    seed INTEGER,
                    top1_correct BOOLEAN,
                    top2_correct BOOLEAN,
                    detection_latency REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to initialize SQLite database at '{db_path}': {e}") from e


def save_readings(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """Save sensor readings from a DataFrame into the 'readings' table.

    Args:
        df: DataFrame containing 'time' and sensor columns (e.g. S1, S2, S3, S4, S5).
        db_path: Path to SQLite database file.

    Returns:
        Total number of sensor rows inserted.
    """
    init_db(db_path)
    time_col = "time" if "time" in df.columns else ("timestep" if "timestep" in df.columns else None)
    if time_col is None:
        raise ValueError("DataFrame must contain a 'time' or 'timestep' column.")

    sensor_cols = [c for c in df.columns if c != time_col and not c.startswith("_")]

    # Build list of (time, sensor, value) tuples for fast parameterized batch insertion
    rows_to_insert = []
    for _, row in df.iterrows():
        t = int(row[time_col])
        for sensor in sensor_cols:
            val = float(row[sensor])
            rows_to_insert.append((t, sensor, val))

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO readings (time, sensor, value) VALUES (?, ?, ?)",
                rows_to_insert,
            )
            conn.commit()
            return len(rows_to_insert)
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error while saving readings to '{db_path}': {e}") from e


def save_report(report_dict: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    """Save a fault localization or evaluation report into 'fault_reports' table.

    Args:
        report_dict: Dictionary containing report data to be stored as JSON.
        db_path: Path to SQLite database file.

    Returns:
        The auto-generated row id of the inserted report.
    """
    init_db(db_path)
    report_json_str = json.dumps(report_dict, default=str)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO fault_reports (report_json) VALUES (?)",
                (report_json_str,),
            )
            conn.commit()
            return cursor.lastrowid or 0
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error while saving report to '{db_path}': {e}") from e


def load_latest_report(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve and parse the most recent report from 'fault_reports' table.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Dictionary parsed from report JSON (including id and created_at), or None if table is empty.
    """
    init_db(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, created_at, report_json FROM fault_reports ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row is None:
                return None
            report_id, created_at, report_json_str = row
            data = json.loads(report_json_str)
            if isinstance(data, dict):
                data["_report_id"] = report_id
                data["_created_at"] = created_at
            return data
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error while loading latest report from '{db_path}': {e}") from e


# Backward compatibility wrapper class for pipeline and tests
class StorageManager:
    """Object-oriented wrapper around SQLite storage functions."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        init_db(self.db_path)

    def save_readings(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> int:
        return save_readings(df, db_path=self.db_path)

    def save_report(self, report_dict: Dict[str, Any]) -> int:
        return save_report(report_dict, db_path=self.db_path)

    def load_latest_report(self) -> Optional[Dict[str, Any]]:
        return load_latest_report(db_path=self.db_path)

    def get_readings_by_run(self, run_id: Optional[str] = None) -> pd.DataFrame:
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM readings", conn)
