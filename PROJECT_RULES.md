# FaultMesh AI — Project Specifications and Rules

This document outlines the architectural specifications, baseline configurations, fault simulation rules, detection requirements, localization algorithms, and evaluation protocol for **FaultMesh AI: Intelligent Cooperative IoT Fault Localization System**.

---

## 1. Sensor Configuration

The IoT sensor network monitors five interconnected sensors with the following baseline operational values:

| Sensor ID | Monitored Parameter | Baseline Value | Units | Description |
|:---:|:---|:---:|:---:|:---|
| **S1** | Temperature | 60.0 | °C | Thermal sensor measuring operating heat |
| **S2** | Vibration | 2.0 | mm/s | Mechanical vibration transducer |
| **S3** | Pressure | 5.0 | bar | Primary fluid/hydraulic pressure sensor |
| **S4** | Flow | 30.0 | L/min | Fluid circulation flow meter |
| **S5** | Current | 10.0 | A | Motor/electrical drive current draw |

---

## 2. Sensor Network Topology

The sensors form an undirected graph $G = (V, E)$ where physical coupling, fluid dynamics, or electrical interactions allow faults to propagate between neighboring nodes.

- **Nodes ($V$):** `{"S1", "S2", "S3", "S4", "S5"}`
- **Edges ($E$):**
  - **S3 – S1** (Pressure – Temperature coupling)
  - **S3 – S2** (Pressure – Vibration coupling)
  - **S3 – S4** (Pressure – Flow coupling)
  - **S1 – S2** (Temperature – Vibration coupling)
  - **S4 – S5** (Flow – Current coupling)
  - **S2 – S5** (Vibration – Current coupling)
  - **S3 – S5** (Pressure – Current coupling)

---

## 3. Simulation Specifications

1. **Sensor Streams:**
   - Generate five continuous time-series streams corresponding to S1 through S5.
2. **Measurement Noise:**
   - Gaussian noise $\mathcal{N}(0, \sigma^2)$ is added to each sensor stream.
   - Standard deviation $\sigma$ is approximately **1.5%** of each sensor's nominal baseline:
     - S1: $\sigma = 0.9$ °C
     - S2: $\sigma = 0.03$ mm/s
     - S3: $\sigma = 0.075$ bar
     - S4: $\sigma = 0.45$ L/min
     - S5: $\sigma = 0.15$ A
3. **Fault Types:**
   - **Drift:** Gradual ramp/drift away from nominal value over time ($x_t = x_t + \alpha \cdot t$).
   - **Spike:** Sudden, short-duration high-amplitude transient anomalies.
   - **Stuck:** Sensor reading freezes at a constant value (variance approaches zero).
4. **Neighbor Coupling & Propagation:**
   - Physical coupling coefficient: **40% (0.40)** of anomalous deviation propagates to directly connected neighbors in the network.
   - Propagation delay: **2 time steps** before neighboring sensors reflect the induced deviation.

---

## 4. Detection Specifications

1. **Algorithm:**
   - **Isolation Forest** (`sklearn.ensemble.IsolationForest`) trained on normal operation baseline.
2. **Feature Engineering:**
   - **Rolling Mean:** Local average across a rolling window (e.g., window size = 10).
   - **Rolling Standard Deviation:** Local volatility measure.
   - **First Difference ($\Delta x_t = x_t - x_{t-1}$):** Rate of change indicator.
   - **Z-Score:** Standardized deviation relative to normal baseline mean and standard deviation.
3. **Baseline Calibration:**
   - First **100 normal samples** are reserved for baseline statistics and detector initialization.
4. **Alarm Triggering Logic:**
   - Initial anomaly-score threshold: **0.6**.
   - Persistence filter: Requires **3 consecutive abnormal readings** exceeding the threshold before triggering an active detection alarm (avoids false positives from transient noise).

---

## 5. Localization Specifications

Fault localization identifies the root-cause sensor from among all sensors exhibiting anomalous behavior, using the graph topology and temporal signatures.

1. **Ranking Candidates:**
   - Rank all suspected faulty sensors by an aggregate fault likelihood score $S_{\text{fault}} \in [0, 1]$.
2. **Initial Scoring Weights:**
   - **Own Anomaly Score (35% / 0.35):** Magnitude of abnormal deviation detected on the node itself.
   - **Detection Earliness (30% / 0.30):** How early the node triggered an alarm compared to others (root causes trigger earlier than coupled neighbors).
   - **Neighbor Support (20% / 0.20):** Fraction of topological neighbors that also manifest anomalies following this node's alert.
   - **Coverage (15% / 0.15):** Network centrality and topological influence across the sensor mesh.

---

## 6. Evaluation Specifications

1. **Benchmark Matrix:**
   - **5 Sensors:** S1, S2, S3, S4, S5 (as injected ground-truth root causes).
   - **3 Fault Types:** Drift, Spike, Stuck.
   - **5 Random Seeds:** 42, 101, 2024, 777, 999.
   - **Total Scenarios:** $5 \times 3 \times 5 = 75$ evaluation runs.
2. **Evaluation Metrics:**
   - **Top-1 Accuracy:** Percentage of runs where the true root-cause sensor is ranked #1.
   - **Top-2 Accuracy:** Percentage of runs where the true root-cause sensor is in the top 2 ranked candidates.
   - **Mean Detection Latency:** Average time steps between fault injection and initial alarm detection.
