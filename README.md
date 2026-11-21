# FaultMesh AI: Intelligent Cooperative IoT Fault Localization System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FaultMesh AI is an intelligent, graph-guided anomaly detection and root-cause localization framework for interconnected IoT sensor networks. In complex industrial IoT meshes, a fault occurring at one sensor often cascades into physical or electrical deviations across coupled neighbors. FaultMesh AI untangles cascading anomalies to isolate the true root cause with high precision.

---

## 📁 Project Structure

```text
FaultMesh_AI/
│
├── PROJECT_RULES.md         # Specifications, baseline configs, and rules
├── README.md                # Project documentation and guide
├── requirements.txt         # Core dependencies
├── .gitignore               # Version control ignore list
├── app.py                   # Main Streamlit web dashboard
├── evaluate.py              # 75-scenario evaluation benchmark suite
├── run_sim_demo.py          # Standalone sensor simulation demonstration
├── run_detect_demo.py       # Standalone anomaly detection demonstration
├── run_localize_demo.py     # Standalone fault localization demonstration
│
├── faultmesh/               # Core Python package
│   ├── __init__.py          # Package initialization and exports
│   ├── simulator.py         # Sensor time-series generation & fault injection
│   ├── processing.py        # Rolling statistics & feature extraction
│   ├── detector.py          # Isolation Forest anomaly detection
│   ├── graph.py             # NetworkX IoT sensor topology management
│   ├── localizer.py         # Multi-criteria candidate ranking algorithm
│   ├── pipeline.py          # End-to-end orchestration pipeline
│   └── storage.py           # SQLite persistence for readings, alerts, & reports
│
├── tests/                   # Automated test suite
│   ├── __init__.py          # Test package marker
│   └── test_scenarios.py    # Unit & scenario tests for all components
│
└── data/                    # Local storage directory
    └── .gitkeep             # Git keep file
```

---

## 🧩 Module Responsibilities

| File | Primary Responsibility |
|---|---|
| `simulator.py` | Generate simulated sensor readings (S1–S5), inject faults (`drift`, `spike`, `stuck`), and propagate neighbor coupling with delay. |
| `processing.py` | Calculate rolling statistics (rolling mean, rolling std, first difference, Z-score) for anomaly detection. |
| `detector.py` | Isolation Forest anomaly detection with baseline calibration (first 100 samples) and consecutive alert persistence filtering. |
| `graph.py` | Build and manage the sensor network graph using NetworkX (nodes, edges, degree centrality, coverage). |
| `localizer.py` | Rank suspected faulty sensors using multi-criteria weighted fusion (own anomaly score, detection earliness, neighbor support, coverage). |
| `pipeline.py` | Connect simulation, detection, and localization into a clean, reusable orchestration workflow. |
| `storage.py` | Store readings, alerts, and evaluation reports in SQLite database. |
| `app.py` | Main interactive Streamlit dashboard for real-time monitoring, simulation controls, and visual diagnosis. |
| `evaluate.py` | Benchmark fault localization over 75 scenarios (5 sensors × 3 fault types × 5 seeds) measuring Top-1 & Top-2 accuracy. |
| `run_sim_demo.py` | Independently test and verify the sensor simulator. |
| `run_detect_demo.py` | Independently test and verify feature processing and anomaly detection. |
| `run_localize_demo.py` | Independently test and verify graph topology and localization ranking. |
| `test_scenarios.py` | Pytest verification of all core system components. |

---

## ⚙️ Sensor Configuration & Network Topology

### Sensors and Baselines

- **S1 (Temperature):** 60.0 °C (Noise: ~1.5% = 0.9 °C)
- **S2 (Vibration):** 2.0 mm/s (Noise: ~1.5% = 0.03 mm/s)
- **S3 (Pressure):** 5.0 bar (Noise: ~1.5% = 0.075 bar)
- **S4 (Flow):** 30.0 L/min (Noise: ~1.5% = 0.45 L/min)
- **S5 (Current):** 10.0 A (Noise: ~1.5% = 0.15 A)

### Mesh Coupling

Coupled edges: **S3–S1**, **S3–S2**, **S3–S4**, **S1–S2**, **S4–S5**, **S2–S5**, **S3–S5**.  
Physical propagation: **40% coupling** transmitted to neighbors after a **2 time-step delay**.

---

## 🚀 Getting Started

### 1. Installation

Create and activate a virtual environment, then install required dependencies:

```bash
# Optional: create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt
```

### 2. Running Standalone Demonstrations

Test individual components independently:

```bash
# 1. Test sensor simulator:
python run_sim_demo.py

# 2. Test anomaly detector:
python run_detect_demo.py

# 3. Test graph-based localization:
python run_localize_demo.py
```

### 3. Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest tests/test_scenarios.py -v
```

### 4. Running the Systematic Evaluation Benchmark

Execute the 75-scenario evaluation matrix:

```bash
python evaluate.py
```

### 5. Launching the Interactive Web Dashboard

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

---

## 📋 Project Specifications

For complete technical specifications, mathematical scoring formulations, and baseline configurations, see [PROJECT_RULES.md](file:///c:/Users/gagan/Desktop/faultmesh-ai/PROJECT_RULES.md).
