# FaultMesh AI 2.0: Explainable Predictive Maintenance & Intelligent Fault Investigation Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.0.0-teal.svg)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-44%20passed-brightgreen.svg)](tests/)

FaultMesh AI 2.0 is an intelligent, graph-guided anomaly detection, root-cause localization, explainable investigation, and predictive maintenance platform for coupled industrial IoT sensor networks. In complex industrial machinery, a fault occurring at one sensor cascades through physical or thermal coupling into connected peers. FaultMesh AI 2.0 untangles cascading deviations to isolate the originating root cause, delivers transparent evidence chains, tracks sensor operational health, and generates physics-informed predictive maintenance protocols.

---

## 🌟 Key Capabilities in FaultMesh AI 2.0

1. **Multi-Sensor Physical Simulation & Coupling Dynamics:**
   - Simulates 5 coupled industrial sensors (Temperature, Vibration, Pressure, Flow, Current) with Gaussian baseline noise.
   - Models physical propagation along mesh coupling edges with fractional attenuation (40%) and time delay ($\Delta t = 2$).
   - Injects realistic industrial fault modalities: continuous linear drift, transient impulsive spikes, and frozen transducer lockup.

2. **Isolation Forest Detection & Onset Timing:**
   - Unsupervised Isolation Forest ensembles trained strictly on healthy normal baseline samples ($t = 0 \dots 99$).
   - Multi-feature extraction: rolling mean, rolling standard deviation, first difference, and standardized Z-score.
   - Calibrated $[0, 1]$ anomaly scores with consecutive-reading alarm onset verification filters.

3. **Graph-Guided Cooperative Root-Cause Localization:**
   - Topological sensor mesh modeled via NetworkX.
   - Cooperative multi-criteria fusion: own anomaly intensity (35%), detection earliness (30%), neighbor propagation support (20%), and topological coverage (15%).
   - Strict ground-truth isolation: zero label leakage to detection or localization algorithms.

4. **FEATURE A: Explainable Fault Investigation Engine (`faultmesh/explainer.py`):**
   - Decomposes diagnostic reasoning into transparent evidence chains (timing precedence $\Delta t$, anomaly intensity, downstream propagation delay, and graph coverage).
   - Distinguishes direct telemetry observations from structural system assumptions.
   - Evaluates confidence using an explicit, defensible rubric (`HIGH`, `MODERATE`, `LOW`, `INDETERMINATE`) rather than treating composite rankings as calibrated probabilities.

5. **FEATURE B: Advanced Sensor Health Monitoring (`faultmesh/health.py`):**
   - Quantitative sensor health score ($0.0 \dots 100.0$) evaluating baseline drift, anomaly frequency, signal freeze, and sustained fault runs.
   - Transparent operational categorization: `HEALTHY` (85-100), `WARNING` (65-84), `DEGRADED` (40-64), `CRITICAL` (0-39), and `INSUFFICIENT DATA`.
   - Actionable list of specific degradation root factors per sensor node.

6. **FEATURE C: Predictive Maintenance Recommendation Engine (`faultmesh/maintenance.py`):**
   - Physics-informed, rule-guided maintenance protocols customized per sensor parameter (S1..S5) and fault signature (drift, spike, stuck).
   - Prioritized urgency categorization (`IMMEDIATE`, `SCHEDULED`, `PREVENTIVE`).
   - Actionable procedure cards with required tools, affected components, and strict safety cautions.
   - Interactive technician verification checklists.
   - Grounded engineering disclaimer: Rule-based maintenance guidance, transparently disclaiming uncalibrated Remaining Useful Life (RUL) claims.

7. **FEATURE D: Advanced Benchmark & Reliability Evaluation (`evaluate.py`):**
   - Full 75-scenario systematic evaluation matrix (5 sensors × 3 fault types × 5 seeds).
   - Parameterized sweeps across Gaussian noise ratios ($1.0\% \dots 3.5\%$), fault magnitude ratios ($25\% \dots 100\%$), and coupling strengths ($20\% \dots 60\%$).
   - Complete detection performance metrics: Precision, Recall, F1-Score, False Positive Rate (FPR) on clean normal streams, and detection latency.

8. **FEATURE E: Intelligent Fault Investigation Reports (`faultmesh/reporting.py`):**
   - Comprehensive multi-stage diagnostic summaries.
   - Structured diagnostic CSV table exports.
   - Publication-grade PDF engineering reports generated via ReportLab featuring banners, executive summaries, localization tables, fleet health assessments, technician checklists, and compliance notices.

9. **FEATURE F: Live IoT Mode — MQTT Telemetry Ingestion (`faultmesh/ingestion/`):**
   - Continuous real-time multi-sensor telemetry ingestion via MQTT (Paho-MQTT client).
   - Strict data validation: sensor ID verification (S1-S5), numeric bounds, engineering unit consistency, and ISO-8601 timestamps.
   - Rolling time-series buffers with chronological ordering, deduplication, and stale-sensor detection.
   - Mandatory baseline calibration workflow: collects verified normal operating data with authorized operator confirmation before AI fault inference activates.
   - Bundled Software Gateway Simulator for development, testing, and fault-injection demonstrations without physical hardware.

---

## 📁 Project Architecture

```text
faultmesh-ai/
│
├── PROJECT_RULES.md            # Technical specifications, formulas, and baseline parameters
├── README.md                   # Comprehensive project documentation
├── requirements.txt            # Python dependencies (includes reportlab)
├── evaluate.py                 # 75-scenario benchmark & extended sensitivity evaluation suite
├── app.py                      # Root Streamlit launcher
│
├── dashboard/                  # Streamlit industrial monitoring interface
│   ├── app.py                  # Main dashboard application entry point & session state
│   ├── components.py           # Reusable UI cards, Plotly charts, graph views, & tables
│   ├── layout.py               # Modular 11-section dashboard layout orchestration
│   ├── utils.py                # Pipeline adapters, caching, and evaluation loaders
│   └── styles.css              # Custom styling definitions
│
├── faultmesh/                  # Core FaultMesh AI 2.0 analytical package
│   ├── __init__.py             # Package exports (version 2.0.0)
│   ├── simulator.py            # Multi-sensor IoT simulation & fault injection
│   ├── processing.py           # Rolling statistics & feature extraction
│   ├── detector.py             # Isolation Forest anomaly scoring & onset detection
│   ├── graph.py                # NetworkX sensor mesh topology management
│   ├── localizer.py            # Multi-criteria weighted root-cause ranking
│   ├── pipeline.py             # Reusable orchestration pipeline
│   ├── storage.py              # SQLite database persistence
│   ├── explainer.py            # Explainable Fault Investigation Engine (Feature A)
│   ├── health.py               # Sensor Health Monitoring Engine (Feature B)
│   ├── maintenance.py          # Predictive Maintenance Advisor (Feature C)
│   └── reporting.py            # Investigation reporting, CSV & PDF generation (Feature E)
│
├── tests/                      # Automated test suite (44 passing tests)
│   ├── test_scenarios.py       # Core pipeline & integration tests (24 tests)
│   ├── test_explainer.py       # Explainability engine tests (5 tests)
│   ├── test_health.py          # Sensor health monitoring tests (5 tests)
│   ├── test_maintenance.py     # Predictive maintenance advisor tests (4 tests)
│   ├── test_reporting.py       # Report synthesis, CSV & PDF export tests (3 tests)
│   └── test_evaluation_extended.py # Extended metrics, precision, recall & sweeps (3 tests)
│
└── data/                       # Local artifacts & evaluation results
    ├── evaluation_results.csv  # Benchmark CSV dataset
    └── evaluation_report.txt   # Benchmark text summary
```

---

## ⚙️ Sensor Baselines & Network Topology

| Sensor | Monitored Parameter | Baseline Value | Expected Noise σ (1.5%) | Physical Subsystem |
|---|---|---|---|---|
| **S1** | Temperature | 60.0 °C | 0.90 °C | Bearing Assembly & Housing |
| **S2** | Vibration | 2.0 mm/s | 0.030 mm/s | Rotating Drive Shaft |
| **S3** | Pressure | 5.0 bar | 0.075 bar | Cooling Manifold |
| **S4** | Flow | 30.0 L/min | 0.450 L/min | Coolant Delivery Piping |
| **S5** | Current | 10.0 A | 0.150 A | Main Drive Motor Phase |

### Mesh Coupling Connections
Physical coupling channels are modeled as undirected graph edges:
- **S3** is coupled to **S1, S2, S4, S5** (hydraulic hub)
- **S1** is coupled to **S2, S3**
- **S2** is coupled to **S1, S3, S5**
- **S4** is coupled to **S3, S5**
- **S5** is coupled to **S2, S3, S4**

Coupled dynamics: `coupled_dev = 0.40 * (src_dev / src_baseline) * peer_baseline` with a 2-step temporal delay.

---

## 🖥️ Streamlit Dashboard Sections (11 Tabs)

1. **Overview:** High-level KPI cards (Monitored Sensors, Total Time Steps, Sensors with Onset, Top Root Cause Suspect).
2. **Sensor Monitoring:** Interactive Plotly time-series charts with channel filtering and baseline references.
3. **Network Topology:** Interactive NetworkX graph view showing highlighted suspects, alarmed peers, and nominal nodes.
4. **Fault Injection:** Visual parameters of injected fault mode (drift, spike, stuck), start timestep, and physical offset.
5. **Anomaly Detection:** Onset timing table identifying earliest alarms and downstream delay intervals.
6. **Root-Cause Localization:** Multi-criteria ranking table displaying composite scores and criterion breakdowns.
7. **Explainable Investigation (NEW):** Decomposed evidence chains, confidence rubrics, neighbor cascades, and observed vs assumed tables.
8. **Sensor Health (NEW):** Fleet Health Index, operational health score cards (0-100), status badges, and degradation driver details.
9. **Predictive Maintenance (NEW):** Targeted technician procedures, equipment lists, safety precautions, and checklists.
10. **Benchmark Evaluation:** Detailed Top-1 and Top-2 accuracy across 75 scenarios, fault types, and sensor breakdowns.
11. **Investigation Reports (NEW):** One-click download of publication-grade PDF investigation reports, diagnostic summary CSVs, and raw telemetry data.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository and navigate to workspace
cd faultmesh-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (44 Tests)

```bash
python -m pytest tests/ -v
```

### 3. Run Benchmark Evaluations

```bash
# Execute standard 75-scenario evaluation matrix:
python evaluate.py
```

### 4. Launch the Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Live IoT Mode & MQTT Broker Setup

FaultMesh AI 2.0 supports real-time industrial telemetry streaming via MQTT.

#### MQTT Telemetry JSON Format:
```json
{
  "machine_id": "MACHINE_01",
  "sensor_id": "S1",
  "value": 61.5,
  "unit": "°C",
  "timestamp": "2026-11-21T16:30:00Z"
}
```

#### Supported Sensors & Units:
- **S1:** Bearing Temperature (`°C`)
- **S2:** Shaft Vibration (`mm/s`)
- **S3:** Cooling Pressure (`bar`)
- **S4:** Coolant Flow (`L/min`)
- **S5:** Motor Current (`A`)

#### Running an MQTT Broker:
1. **Local Mosquitto (Recommended):**
   ```bash
   # Linux / macOS
   mosquitto -p 1883
   
   # Windows (via winget or installer)
   winget install EclipseFoundation.Mosquitto
   net start mosquitto
   ```
2. **Docker:**
   ```bash
   docker run -d -p 1883:1883 eclipse-mosquitto
   ```
3. **Public Test Broker (for testing):**
   - Host: `broker.hivemq.com`
   - Port: `1883`
   - Topic: `faultmesh/your_unique_id/sensors/#`

#### Software Gateway Simulator:
Included directly in the Live IoT Mode sidebar. It publishes simulated industrial telemetry packets using the physics-grounded `SensorSimulator` to evaluate normal and fault conditions (drift, spike, stuck) without requiring physical plant machinery.

---

## ⚖️ Engineering & Scientific Disclaimers

1. **Simulated Telemetry:** Sensor telemetry in Simulation Mode and Software Gateway Simulator is synthesized using physical modeling equations with Gaussian noise distributions and neighbor coupling dynamics.
2. **Read-Only Telemetry Ingestion:** The Live IoT ingestion architecture operates strictly in read-only telemetry monitoring mode. It does not issue physical actuator commands or modify plant control logic.
3. **Predictive Maintenance Guidance:** Maintenance protocols and technician checklists represent rule-guided diagnostic procedures derived from detected anomaly patterns. They provide actionable maintenance recommendations but do **not** claim empirical Remaining Useful Life (RUL) predictions without historical run-to-failure runout data.
4. **Ground-Truth Separation:** Ground-truth fault injection metadata is strictly isolated for evaluation verification and is never exposed to the Isolation Forest detector or the graph localization algorithm.
