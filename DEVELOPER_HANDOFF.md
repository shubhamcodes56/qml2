# QML TB Detection — Developer Handoff Document

> **Project:** Quantum Machine Learning for Early TB Detection  
> **Purpose:** A clinical decision-support dashboard that compares Classical ML vs QML (8-Qubit Variational Circuit) for detecting Tuberculosis from patient vitals + X-ray images.  
> **Key Thesis:** QML detects "biological sync breakdown" (via Von Neumann Entropy) BEFORE classical thresholds trigger, enabling earlier intervention.

---

## 1. Architecture Overview

```
Frontend (React + Vite + Recharts + TailwindCSS)
    │
    │  Polls /live-history every 1s
    │  POST /analyze-patient (static)
    │  POST /upload-xray (static)
    │  POST /upload-xray-live (live injection)
    │
    ▼
Backend (FastAPI + PennyLane + scikit-learn + OpenCV)
    │
    ├── api.py ─── Main server + simulation loop thread
    ├── quantum_tb_engine.py ─── 8-Qubit QML circuit
    ├── icu_simulator.py ─── Live patient vitals generator
    ├── xray_analyzer.py ─── OpenCV X-ray feature extraction
    ├── generate_tb_data.py ─── Synthetic dataset (16 features, 7 tiers)
    └── sklearn RandomForest ─── Classical ML baseline
```

---

## 2. Backend Files — What Each Does

### Core Files (MUST KEEP)

| File | Purpose |
|------|---------|
| `api.py` | **Main server.** FastAPI app with all endpoints. Runs a background `simulation_loop` thread that generates live patient data every 1 second. |
| `quantum_tb_engine.py` | **THE QML MODEL.** 8-qubit, 4-layer variational circuit using PennyLane. Takes 16 features → encodes into 256-dim Hilbert space → computes Von Neumann Entropy → outputs risk score. |
| `icu_simulator.py` | **Live patient simulator.** Generates realistic vital signs (HR, SpO2, RR, Temp) with biological noise. Has `trigger_anomaly()` to inject micro-drift. |
| `xray_analyzer.py` | **X-Ray feature extraction.** Uses OpenCV to extract opacity, cavity, nodule, pleural scores + 9-zone grid analysis with cropped heatmap images. |
| `generate_tb_data.py` | **Dataset generator.** Creates medically-accurate TB patient records with 16 features, 7 severity tiers, age/gender/comorbidity stratification. Used to train the Classical RF baseline at startup. |

### Training & Evaluation Files (Run Once)

| File | Purpose |
|------|---------|
| `train_tb.py` | Trains the QML circuit weights. Outputs `saved_model/tb_weights.npy`. |
| `train.py` | Alternative/older training script. |
| `compare_models.py` | Runs Classical vs QML comparison and generates metrics. |
| `classical_baseline.py` | Standalone classical ML training and eval. |

### Supporting Files (Nice to Have)

| File | Purpose |
|------|---------|
| `qdrant_store.py` | Vector DB store for patient similarity search using quantum state embeddings. |
| `quantum_vitals.py` | Simpler 4-qubit vitals-only quantum engine (older version). |
| `generate_html_report.py` | Generates standalone HTML report for a patient. |
| `feature_extractor.py` | Advanced feature extraction pipeline. |

---

## 3. The 16 Features (Input Vector)

Every analysis takes a 16-dimensional float vector in this exact order:

```python
FEATURE_NAMES = [
    # A. Core Vitals (4)
    'heart_rate',       # 0  | bpm        | Normal: 60-100
    'spo2',             # 1  | %          | Normal: 95-100
    'resp_rate',        # 2  | breaths/min| Normal: 12-20
    'temperature',      # 3  | C          | Normal: 36.1-37.2

    # B. Blood Markers (6)
    'wbc_count',        # 4  | x10^3/uL   | Normal: 4.5-11.0
    'esr',              # 5  | mm/hr      | Normal: 0-20
    'crp',              # 6  | mg/L       | Normal: 0-5
    'lymphocyte_pct',   # 7  | %          | Normal: 20-40
    'hemoglobin',       # 8  | g/dL       | Normal: 12-17
    'albumin',          # 9  | g/dL       | Normal: 3.5-5.5

    # C. X-Ray Scores (4) — extracted by xray_analyzer.py
    'xray_opacity',     # 10 | 0-1 score  | Normal: <0.10
    'xray_cavity',      # 11 | 0-1 score  | Normal: <0.05
    'xray_nodule',      # 12 | 0-1 score  | Normal: <0.08
    'xray_pleural',     # 13 | 0-1 score  | Normal: <0.06

    # D. TB-Specific Tests (2)
    'ada_level',        # 14 | U/L        | Normal: <40
    'mantoux_mm',       # 15 | mm         | Normal: <10
]
```

---

## 4. API Endpoints

### Live Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /live-history` | GET | Returns sliding window (last 120s) of live simulation data. Frontend polls this every 1s. |
| `POST /trigger-drift` | POST | Injects micro-drift anomaly into the live patient. Simulates early TB onset. |
| `POST /reset` | POST | Resets simulation to healthy baseline. |
| `POST /upload-xray-live` | POST (multipart) | Upload X-ray; its features get injected into the LIVE patient state. |

### Static Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /analyze-patient` | POST (JSON) | Send full 16-feature patient vitals for one-time Classical + QML analysis. |
| `POST /upload-xray` | POST (multipart) | Upload X-ray for standalone QML analysis (static tab). |

### Response Schemas

#### `GET /live-history` Response
```json
{
  "history": [
    {
      "time_index": 42,
      "timestamp": 1724853912.123,
      "is_drift": false,
      "classical_vitals": {
        "heart_rate": 76.2,
        "temperature": 36.98,
        "spo2": 97.8
      },
      "qml_correlations": {
        "hr_temp_sync": 0.012,
        "spo2_wbc_coupling": -0.034,
        "entanglement_entropy": 3.02
      },
      "risk": {
        "classical": 18.6,
        "qml": 4.0
      },
      "annotations": []
    }
  ]
}
```

#### `POST /analyze-patient` Request
```json
{
  "vitals": {
    "heart_rate": 78.0,
    "spo2": 97.5,
    "resp_rate": 16.0,
    "temperature": 36.8,
    "wbc_count": 7.0,
    "esr": 10.0,
    "crp": 2.0,
    "lymphocyte_pct": 30.0,
    "hemoglobin": 13.5,
    "albumin": 4.0,
    "xray_opacity": 0.08,
    "xray_cavity": 0.04,
    "xray_nodule": 0.06,
    "xray_pleural": 0.05,
    "ada_level": 15.0,
    "mantoux_mm": 5.0
  }
}
```

#### `POST /analyze-patient` Response
```json
{
  "classical_ml": {
    "risk_score": 0.186,
    "diagnosis": "Healthy",
    "features": [
      {"name": "heart_rate", "value": 78.0, "importance": 0.12, "status": "normal"}
    ]
  },
  "quantum_ml": {
    "risk_score": 4,
    "severity": "NORMAL",
    "von_neumann_entropy": 3.02,
    "mutual_information": {
      "entropy_vitals": 2.91,
      "entropy_blood_xray": 2.85,
      "mutual_information": 5.76
    },
    "circuit_angles": [0.0, 0.0],
    "density_matrix_heatmap": [[]],
    "probability_landscape": [[]],
    "n_qubits": 8,
    "hilbert_dim": 256
  },
  "advisory": []
}
```

#### `POST /upload-xray` Response
```json
{
  "xray_analysis": {
    "opacity_score": 0.318,
    "cavity_probability": 0.012,
    "nodule_density": 0.15,
    "pleural_thickening": 0.465,
    "zones": [
      {
        "zone_id": 1, "row": 0, "col": 0,
        "mean_intensity": 0.33, "std_intensity": 0.12,
        "max_intensity": 0.85, "dark_ratio": 0.28,
        "zone_image_b64": "iVBOR..."
      }
    ],
    "heatmap_overlay_b64": "iVBOR...",
    "findings": [
      {
        "feature": "Lung Opacity",
        "value": 0.318,
        "severity": "warning",
        "message": "Minor diffuse opacity patterns detected in zones 8.",
        "recommendation": "Consider CT follow-up if persistent >48hrs"
      }
    ]
  },
  "qml_with_xray": {},
  "advisory": []
}
```

---

## 5. How the QML Risk Score Works

```
Patient Vitals (16 features)
    |
    v
Normalize to angles [-pi, pi]
    |
    v
+-------------------------------------+
|  8-Qubit Variational Circuit        |
|                                     |
|  Layer 0: Encode vitals (RY)        |
|  Layer 1: Encode blood/xray (RY)    |
|  Layer 2: Re-upload ALL (RY+RZ)     |
|  Layer 3: Cross-domain mix          |
|                                     |
|  Each layer has:                    |
|  - Data encoding rotations          |
|  - Trainable Rot() gates            |
|  - Ring + Skip + Cross-group CNOTs  |
+-------------------------------------+
    |
    v
256-dim State Vector |psi>
    |
    v
Bipartition: Qubits 0-3 (vitals) | Qubits 4-7 (blood/xray)
    |
    v
Von Neumann Entropy S = -Tr(rho_A * log2(rho_A))
    |
    v
Risk = sigmoid(|S - 3.02| - 0.15)
    |
    v
Healthy S ~ 3.02 --> Risk 4%
Micro-drift S ~ 2.70 --> Risk 96%
Severe TB S ~ 3.29 --> Risk 91%
```

**Key insight for the doctor:** When a patient is healthy, the "biological sync" between vitals (heart, lungs) and blood markers (WBC, CRP) stays at a stable entropy level (~3.02). Any infection, even before symptoms appear, disrupts this sync. QML measures this disruption as entropy deviation. Classical ML only checks if individual values cross fixed thresholds.

---

## 6. Frontend Components

| Component | File | What It Renders |
|-----------|------|-----------------|
| **App Shell** | `App.tsx` | Tab navigation (Live Timeline / Static Analysis), header |
| **TimelineView** | `TimelineView.tsx` | 3 live-updating Recharts graphs: Classical Vitals, QML Correlations, Risk Score comparison. Polls `/live-history` every 1s. Has "Inject Micro-Drift", "Reset", "Upload X-Ray" buttons. |
| **PatientInput** | `PatientInput.tsx` | Form with all 16 feature inputs for manual static analysis |
| **ClassicalMLView** | `ClassicalMLView.tsx` | Shows Classical RF results: risk score, feature importances bar chart |
| **QMLEntanglementView** | `QMLEntanglementView.tsx` | Shows QML results: density matrix heatmap, probability landscape, entropy metrics |
| **XrayPanel** | `XrayPanel.tsx` | X-ray upload, heatmap overlay display, interactive 9-zone grid with click-to-zoom |
| **AdvisoryPanel** | `AdvisoryPanel.tsx` | Doctor-friendly findings and recommendations |

---

## 7. Setup & Run

### Backend
```bash
cd QML_Report/backend
pip install -r requirements.txt
pip install opencv-python  # For X-ray analysis

# Train the QML model (run once, saves weights to saved_model/)
python train_tb.py

# Start the API server
python api.py
# --> Runs on http://localhost:8000
```

### Frontend
```bash
cd QML_Report/frontend-react
npm install
npm run dev
# --> Runs on http://localhost:5173
```

### Required Files
```
backend/
  saved_model/
    tb_weights.npy    <-- CRITICAL: trained QML weights (run train_tb.py to generate)
```

---

## 8. Key Calibration Values

These values are hardcoded and calibrated. DO NOT change without re-running calibration:

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| `HEALTHY_ENTROPY` | 3.02 | `quantum_tb_engine.py` | Baseline entropy for a perfectly healthy patient |
| `Sigmoid center` | 0.15 | `quantum_tb_engine.py` | Entropy deviation at which risk is approx 50% |
| `Sigmoid steepness` | 20.0 | `quantum_tb_engine.py` | How sharply risk transitions |
| `Entropy alert threshold` | 1.5 + risk>40 | `api.py` | When to show "Entropy Spike" annotation |
| `Cooldown` | 15 seconds | `api.py` | Min gap between entropy spike alerts |
| `MAX_HISTORY` | 120 | `api.py` | Sliding window size for live data |

---

## 9. Healthy vs Drift Patient Profiles

### Healthy Baseline (Tier-0)
```python
base_patient_profile = {
    'wbc_count': 7.0, 'esr': 10.0, 'crp': 2.0, 'lymphocyte_pct': 30.0,
    'hemoglobin': 13.5, 'albumin': 4.0,
    'xray_opacity': 0.08, 'xray_cavity': 0.04, 'xray_nodule': 0.06, 'xray_pleural': 0.05,
    'ada_level': 15.0, 'mantoux_mm': 5.0
}
```

### Drift Targets (When anomaly triggered)
```python
# In icu_simulator.py - these are "normal range but shifted"
target_hr = 88.0      # Still < 100 (normal range)
target_spo2 = 95.0    # Still > 94 (normal range)
target_rr = 22.0      # Slightly elevated
target_temp = 37.8    # Sub-febrile (not flagged by classical)
```

---

## 10. Production Deployment Notes

**IMPORTANT: For production, replace these components:**

1. **ICU Simulator** -> Real patient data feed (HL7/FHIR endpoint, Kafka stream, or hospital EHR API)
2. **RandomForest (in-memory)** -> Persistent model stored on disk or MLflow
3. **PennyLane `default.qubit`** -> For real quantum: `qml.device("qiskit.ibmq", ...)` or keep simulator for speed
4. **OpenCV X-ray analysis** -> Replace with a trained CNN (ResNet/DenseNet) for real clinical-grade X-ray scoring
5. **Polling (`setInterval`)** -> WebSocket for true real-time streaming
6. **In-memory `live_history` list** -> Redis or TimescaleDB for persistence

**WARNING: The QML weights (`tb_weights.npy`) are trained on synthetic data.** For clinical use, retrain on real anonymized TB datasets (e.g., WHO, NTEP). The circuit architecture is production-ready; only the weights need retraining.

---

## 11. File Tree (What to Give Your Developer)

```
QML_Report/
+-- backend/
|   +-- api.py                    * Main server (FastAPI)
|   +-- quantum_tb_engine.py      * QML model (PennyLane 8-qubit)
|   +-- icu_simulator.py          * Live patient simulator
|   +-- xray_analyzer.py          * X-ray feature extraction
|   +-- generate_tb_data.py       * Dataset generator (16 features)
|   +-- train_tb.py               * QML training script
|   +-- requirements.txt          * Python dependencies
|   +-- saved_model/
|   |   +-- tb_weights.npy        * Trained QML weights
|   +-- qdrant_store.py             Vector DB (optional)
|   +-- classical_baseline.py       Classical ML training
|   +-- compare_models.py          Model comparison
|
+-- frontend-react/
|   +-- package.json              * Node dependencies
|   +-- src/
|   |   +-- App.tsx               * Main app shell
|   |   +-- index.css             * Global styles (doctor theme)
|   |   +-- main.tsx              * React entry point
|   |   +-- components/
|   |       +-- TimelineView.tsx   * Live streaming dashboard
|   |       +-- PatientInput.tsx   * Patient form
|   |       +-- XrayPanel.tsx      * X-ray upload + zone zoom
|   |       +-- ClassicalMLView.tsx  Classical results
|   |       +-- QMLEntanglementView.tsx  Quantum results
|   |       +-- AdvisoryPanel.tsx  Clinical advice
|   +-- ...config files
+-- DEVELOPER_HANDOFF.md          * THIS FILE
```

* = Essential files. Everything else is supporting/optional.

---

## 12. Tech Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend Framework | React | 19.x |
| Build Tool | Vite | 8.x |
| Charts | Recharts | 3.x |
| Styling | TailwindCSS | 4.x |
| Backend Framework | FastAPI | latest |
| QML Framework | PennyLane | latest |
| Classical ML | scikit-learn | latest |
| Image Processing | OpenCV (cv2) | latest |
| Vector DB | Qdrant (optional) | latest |
| Dataset | PyTorch tensors | latest |
| Language | Python 3.10+ / TypeScript 6.0 | |
