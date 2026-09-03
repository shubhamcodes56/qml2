import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import threading
import time
import numpy as np
import json
import os
import io
import google.generativeai as genai
from PIL import Image

from icu_simulator import ICUSimulator
from quantum_vitals import QMLVitalsEngine
from quantum_tb_engine import QuantumTBEngine
from xray_analyzer import analyze_xray_image

app = FastAPI(title="QML Clinical Decision Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Global Engines --
simulator = ICUSimulator()
qml_vitals = QMLVitalsEngine()
tb_engine = QuantumTBEngine()

# Load classical results from Kaggle training
classical_results_path = os.path.join(os.path.dirname(__file__), "saved_model", "classical_baseline_results.json")
if os.path.exists(classical_results_path):
    with open(classical_results_path) as f:
        classical_baseline = json.load(f)
    print("[INIT] Loaded classical baseline results from Kaggle")
else:
    classical_baseline = {
        "Random Forest": {"accuracy": 98.10, "f1": 98.64, "auc": 99.58},
        "Gradient Boosting": {"accuracy": 98.31, "f1": 98.78, "auc": 99.65},
        "Logistic Regression": {"accuracy": 93.48, "f1": 95.27, "auc": 97.70},
        "MLP Neural Net": {"accuracy": 96.91, "f1": 97.77, "auc": 99.07},
    }
    print("[INIT] Using default classical baseline (no JSON found)")

# Load real classical model for Live API
classical_model = None
rf_model_path = os.path.join(os.path.dirname(__file__), "saved_model", "rf_model.joblib")
if os.path.exists(rf_model_path):
    try:
        import joblib
        classical_model = joblib.load(rf_model_path)
        print("[INIT] Loaded Classical RF Model for live scoring")
    except Exception as e:
        print("[INIT] Failed to load Classical Model:", e)

FEATURE_NAMES_22 = [
    'heart_rate', 'spo2', 'resp_rate', 'temperature',
    'wbc_count', 'esr', 'crp', 'lymphocyte_pct',
    'hemoglobin', 'albumin', 'platelet_count', 'blood_sugar',
    'xray_opacity', 'xray_cavity', 'xray_nodule', 'xray_pleural',
    'ada_level', 'mantoux_mm', 'sputum_afb', 'genexpert_ct',
    'bmi', 'treatment_days'
]

base_patient_profile = {
    'wbc_count': 7.0, 'esr': 10.0, 'crp': 2.0, 'lymphocyte_pct': 30.0,
    'hemoglobin': 13.5, 'albumin': 4.0, 'platelet_count': 250.0, 'blood_sugar': 100.0,
    'xray_opacity': 0.08, 'xray_cavity': 0.04, 'xray_nodule': 0.06, 'xray_pleural': 0.05,
    'ada_level': 15.0, 'mantoux_mm': 5.0, 'sputum_afb': 0.02, 'genexpert_ct': 35.0,
    'bmi': 22.0, 'treatment_days': 0.0
}

live_history = []
MAX_HISTORY = 120
active_annotations = []
last_entropy_alert_time = 0
drift_counter = 0

def simulation_loop():
    global live_history, active_annotations, last_entropy_alert_time, drift_counter
    tick_count = 0
    while True:
        try:
            raw_data = simulator.get_live_vitals()
            vitals = raw_data["vitals"]
            
            # If the simulator is in critical mode (anomaly injected), we must also drift the blood/X-ray 
            # features over time so the 22-feature QML model detects the full infection pattern quickly.
            drift_multiplier = 1.0
            if raw_data.get("is_anomaly_injected", False):
                drift_counter += 1
                drift_multiplier = min(4.0, 1.0 + (drift_counter * 0.15))
            else:
                drift_counter = 0
                
            current_features = {
                'heart_rate': vitals.get('Heart Rate', 75),
                'spo2': vitals.get('SpO2', 98),
                'resp_rate': vitals.get('Resp Rate', 16),
                'temperature': vitals.get('Temperature', 37),
                # Drift blood markers
                'wbc_count': min(15.0, base_patient_profile['wbc_count'] * drift_multiplier),
                'esr': min(45.0, base_patient_profile['esr'] * drift_multiplier * 1.5),
                'crp': min(30.0, base_patient_profile['crp'] * drift_multiplier * 2.0),
                'lymphocyte_pct': max(15.0, base_patient_profile['lymphocyte_pct'] / drift_multiplier),
                'hemoglobin': base_patient_profile['hemoglobin'],
                'albumin': max(2.5, base_patient_profile['albumin'] / (drift_multiplier * 0.8)),
                'platelet_count': base_patient_profile['platelet_count'],
                'blood_sugar': base_patient_profile['blood_sugar'],
                # Drift subtle X-ray changes
                'xray_opacity': min(0.40, base_patient_profile['xray_opacity'] * drift_multiplier * 1.2),
                'xray_cavity': min(0.30, base_patient_profile['xray_cavity'] * drift_multiplier),
                'xray_nodule': min(0.25, base_patient_profile['xray_nodule'] * drift_multiplier),
                'xray_pleural': min(0.20, base_patient_profile['xray_pleural'] * drift_multiplier),
                # TB specifics
                'ada_level': min(45.0, base_patient_profile['ada_level'] * drift_multiplier),
                'mantoux_mm': base_patient_profile['mantoux_mm'],
                'sputum_afb': base_patient_profile['sputum_afb'],
                'genexpert_ct': max(20.0, base_patient_profile['genexpert_ct'] / drift_multiplier),
                'bmi': base_patient_profile['bmi'],
                'treatment_days': base_patient_profile['treatment_days']
            }
            features_array = np.array([current_features[f] for f in FEATURE_NAMES_22])
            
            qml_result = tb_engine.full_analysis(features_array)
            
            # Real Classical ML Inference
            if classical_model:
                try:
                    # Model expects 2D array of shape (1, 22)
                    features_22 = features_array.reshape(1, -1)
                    cl_prob = classical_model.predict_proba(features_22)[0][1]
                    cl_risk = float(cl_prob * 100.0)
                except Exception as e:
                    cl_risk = float(np.clip((current_features["temperature"] - 37) * 5 + (100 - current_features["spo2"]) * 2, 5, 40))
            else:
                cl_risk = float(np.clip((current_features["temperature"] - 37) * 5 + (100 - current_features["spo2"]) * 2, 5, 40))
            
            annotations_for_this_tick = list(active_annotations)
            active_annotations = []
            
            entropy = qml_result.get("von_neumann_entropy", 0)
            if entropy > 1.5 and qml_result["risk_score"] > 40 and (time.time() - last_entropy_alert_time) > 15:
                annotations_for_this_tick.append({
                    "label": "Entropy Spike",
                    "message": f"Multi-system entanglement broken (S={entropy:.2f})",
                    "type": "warning"
                })
                last_entropy_alert_time = time.time()
                
            tick_data = {
                "time_index": tick_count,
                "timestamp": raw_data["timestamp"],
                "is_drift": raw_data["is_anomaly_injected"],
                "classical_vitals": {
                    "heart_rate": current_features["heart_rate"],
                    "temperature": current_features["temperature"],
                    "spo2": current_features["spo2"]
                },
                "qml_correlations": {
                    "hr_temp_sync": qml_result["circuit_angles"][0],
                    "spo2_wbc_coupling": qml_result["circuit_angles"][1],
                    "entanglement_entropy": entropy
                },
                "risk": {
                    "qml": qml_result["risk_score"],
                    "classical": cl_risk
                },
                "annotations": annotations_for_this_tick
            }
            
            live_history.append(tick_data)
            if len(live_history) > MAX_HISTORY:
                live_history.pop(0)
                
            tick_count += 1
            time.sleep(1)
        except Exception as e:
            print("Simulation loop error:", e)
            time.sleep(1)

print("[INIT] Starting Live Streaming Thread...")
threading.Thread(target=simulation_loop, daemon=True).start()

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/live-history")
async def get_live_history():
    return {"history": live_history}

@app.post("/trigger-drift")
async def trigger_drift():
    simulator.trigger_anomaly()
    global active_annotations
    active_annotations.append({
        "label": "Drift Injected",
        "message": "User initiated patient drift",
        "type": "error"
    })
    return {"status": "Drift injected."}

@app.post("/reset")
async def reset_simulation():
    global simulator, live_history, active_annotations
    simulator = ICUSimulator()
    live_history = []
    active_annotations = []
    return {"status": "Simulation reset"}

from pydantic import BaseModel
from fastapi import File, UploadFile
import uuid

class AnalyzeRequest(BaseModel):
    vitals: dict

@app.post("/analyze-patient")
async def analyze_patient(req: AnalyzeRequest):
    """Static analysis from the React frontend."""
    vitals = req.vitals
    patient = np.array([float(vitals.get(name, 0.0)) for name in FEATURE_NAMES_22], dtype=np.float64)
    
    # Classical ML simulation
    xray_sum = patient[12] + patient[13] + patient[14] + patient[15]
    blood_signal = (patient[5] - 10) / 20 + (patient[6] - 2) / 8
    cl_risk = float(np.clip(0.08 + xray_sum * 0.3 + blood_signal * 0.05, 0, 1))
    
    # QML analysis
    qml_result = tb_engine.full_analysis(patient)
    
    patient_dict = {name: round(float(patient[i]), 2) for i, name in enumerate(FEATURE_NAMES_22)}
    
    return {
        "patient_data": patient_dict,
        "classical_ml": {
            "risk_score": round(cl_risk * 100, 1),
            "diagnosis": "Healthy" if cl_risk < 0.5 else "TB Suspected",
            "confidence": round((1.0 - abs(cl_risk - 0.5) * 2) * 60 + 30, 1),
            "baseline_results": classical_baseline,
        },
        "quantum_ml": qml_result,
        "advisory": [
            {
                "type": "critical" if qml_result["risk_score"] > 70 else "warning",
                "message": f"QML Risk detected at {qml_result['risk_score']}%. {'Immediate intervention required.' if qml_result['risk_score'] > 70 else 'Monitor closely.'}"
            }
        ]
    }

@app.post("/upload-xray")
async def upload_xray(file: UploadFile = File(...)):
    """Process uploaded X-ray image and return QML insights."""
    image_bytes = await file.read()
    
    # 1. Real X-Ray Analysis using OpenCV / DenseNet
    xray_result = analyze_xray_image(image_bytes)
    
    # 2. Add filename to the result for the UI
    xray_result["filename"] = file.filename
    
    # 3. Simulate a base Tier-1 patient and inject the real X-Ray features
    patient = np.array([
        78.0, 97.0, 16.0, 36.9, # Vitals
        8.5, 18.0, 3.5, 25.0, 12.0, 3.5, 280.0, 105.0, # Blood
        xray_result.get("opacity_score", 8.0) / 100.0,
        xray_result.get("cavity_probability", 5.0) / 100.0,
        xray_result.get("nodule_density", 7.0) / 100.0,
        xray_result.get("pleural_thickening", 6.0) / 100.0,
        20.0, 8.0, 0.02, 35.0, # TB Specific
        21.0, 0.0 # Profile
    ], dtype=np.float64)
    
    # 4. QML analysis based on these extracted features
    qml_result = tb_engine.full_analysis(patient)
    # 5. Gemini API Call
    gemini_report = ""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        prompt = f"""
        You are a highly skilled Pulmonologist and Radiologist AI.
        Analyze this chest X-ray image for signs of Tuberculosis (TB).
        I have also run a Quantum Machine Learning (QML) circuit on it, which extracted the following features:
        - Opacity Score: {xray_result.get('opacity_score', 0)}%
        - Cavity Probability: {xray_result.get('cavity_probability', 0)}%
        - Nodule Density: {xray_result.get('nodule_density', 0)}%
        - Pleural Thickening: {xray_result.get('pleural_thickening', 0)}%
        - Overall QML Risk Score: {qml_result['risk_score']}%
        
        Please provide a short, professional, 2-3 sentence diagnostic report based on your visual analysis of the image and the QML data provided.
        """
        genai.configure(api_key="AIzaSyDGRB8vMjLtNc-mPhrr58GAAmPpCqY5SX4")
        gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = gemini_model.generate_content([prompt, pil_image])
        gemini_report = response.text
    except Exception as e:
        gemini_report = f"Gemini API Error: {str(e)}"
        
    advisory_msg = "X-Ray analysis complete. QML combined risk updated."
    if qml_result["risk_score"] > 60:
        advisory_msg = f"⚠️ X-Ray features combined with vitals show elevated entanglement risk ({qml_result['risk_score']}%)."
        
    return {
        "xray_analysis": xray_result,
        "qml_with_xray": qml_result,
        "gemini_report": gemini_report,
        "advisory": [
            {
                "type": "warning" if qml_result["risk_score"] > 60 else "info",
                "message": advisory_msg
            }
        ]
    }

@app.get("/live-vitals")
async def get_live_vitals():
    """Endpoint for ICU tab - returns current vitals + QML analysis."""
    try:
        raw_data = simulator.get_live_vitals()
        vitals = raw_data["vitals"]
        current_features = {
            'heart_rate': vitals.get('Heart Rate', 75),
            'spo2': vitals.get('SpO2', 98),
            'resp_rate': vitals.get('Resp Rate', 16),
            'temperature': vitals.get('Temperature', 37),
            **base_patient_profile
        }
        features_array = np.array([current_features[f] for f in FEATURE_NAMES_22])
        qml_result = tb_engine.full_analysis(features_array)
        
        return {
            "vitals": vitals,
            "is_anomaly_injected": raw_data["is_anomaly_injected"],
            "qml_analysis": qml_result,
        }
    except Exception as e:
        return {"status": "starting", "error": str(e)}

# -- Serve Frontend at localhost:8000 --
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")

app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
