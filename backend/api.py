import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import numpy as np

from icu_simulator import ICUSimulator
from quantum_vitals import QMLVitalsEngine
from quantum_tb_engine import QuantumTBEngine
from xray_analyzer import analyze_xray_image
from generate_tb_data import generate_tb_dataset
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="QML Clinical Decision Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Engines ──
simulator = ICUSimulator()
qml_vitals = QMLVitalsEngine()
tb_engine = QuantumTBEngine()

print("[INIT] Training Classical ML baseline...")
X_train, y_train, _ = generate_tb_dataset(2000)
classical_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
classical_rf.fit(X_train, y_train)

FEATURE_NAMES = [
    'heart_rate', 'spo2', 'resp_rate', 'temperature',
    'wbc_count', 'esr', 'crp', 'lymphocyte_pct',
    'hemoglobin', 'albumin',
    'xray_opacity', 'xray_cavity', 'xray_nodule', 'xray_pleural',
    'ada_level', 'mantoux_mm'
]

# Baseline for non-ICU vitals (Tier-0 Completely Healthy preset)
base_patient_profile = {
    'wbc_count': 7.0, 'esr': 10.0, 'crp': 2.0, 'lymphocyte_pct': 30.0,
    'hemoglobin': 13.5, 'albumin': 4.0,
    'xray_opacity': 0.08, 'xray_cavity': 0.04, 'xray_nodule': 0.06, 'xray_pleural': 0.05,
    'ada_level': 15.0, 'mantoux_mm': 5.0
}

live_history = []
MAX_HISTORY = 120  # Store last 120 seconds of data

# Keep track of active annotations (e.g. from X-ray)
active_annotations = []
last_entropy_alert_time = 0

def simulation_loop():
    global live_history, active_annotations, last_entropy_alert_time
    tick_count = 0
    while True:
        try:
            # 1. Get live ICU vitals
            raw_data = simulator.get_live_vitals()
            vitals = raw_data["vitals"]
            
            # Combine ICU vitals with base profile to get all 16 features
            current_features = {
                'heart_rate': vitals.get('Heart Rate', 75),
                'spo2': vitals.get('SpO2', 98),
                'resp_rate': vitals.get('Resp Rate', 16),
                'temperature': vitals.get('Temperature', 37),
                **base_patient_profile
            }
            features_array = np.array([current_features[f] for f in FEATURE_NAMES])
            
            # 2. Run Classical ML
            cl_prob = float(classical_rf.predict_proba([features_array])[0][1])
            
            # 3. Run QML Engine
            qml_result = tb_engine.full_analysis(features_array)
            
            # 4. Check for dynamic alerts
            annotations_for_this_tick = list(active_annotations)
            active_annotations = [] # clear after appending
            
            entropy = qml_result.get("von_neumann_entropy", 0)
            if entropy > 1.5 and qml_result["risk_score"] > 40 and (time.time() - last_entropy_alert_time) > 15: # Cooldown of 15 ticks
                annotations_for_this_tick.append({
                    "label": "Entropy Spike",
                    "message": f"Multi-system entanglement broken (S={entropy:.2f})",
                    "type": "warning"
                })
                last_entropy_alert_time = time.time()
                
            # 5. Append to history
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
                    "hr_temp_sync": qml_result["circuit_angles"][0], # Approx tracking
                    "spo2_wbc_coupling": qml_result["circuit_angles"][1],
                    "entanglement_entropy": entropy
                },
                "risk": {
                    "classical": cl_prob * 100,
                    "qml": qml_result["risk_score"]
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

# ══════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════

@app.get("/live-history")
async def get_live_history():
    """Returns the sliding window of live simulation data."""
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
    global simulator, live_history, base_patient_profile, active_annotations
    simulator = ICUSimulator()
    live_history = []
    active_annotations = []
    # Reset base profile to default
    base_patient_profile = {
        'wbc_count': 7.0, 'esr': 10.0, 'crp': 2.0, 'lymphocyte_pct': 30.0,
        'hemoglobin': 13.5, 'albumin': 4.0,
        'xray_opacity': 0.08, 'xray_cavity': 0.04, 'xray_nodule': 0.06, 'xray_pleural': 0.05,
        'ada_level': 15.0, 'mantoux_mm': 5.0
    }
    return {"status": "Simulation reset"}

@app.post("/upload-xray-live")
async def upload_xray_live(file: UploadFile = File(...)):
    """Upload chest X-ray and inject its features into the LIVE simulation."""
    global base_patient_profile, active_annotations
    try:
        contents = await file.read()
        xray_result = analyze_xray_image(contents)
        
        # Inject into live patient profile
        base_patient_profile['xray_opacity'] = xray_result["opacity_score"]
        base_patient_profile['xray_cavity'] = xray_result["cavity_probability"]
        base_patient_profile['xray_nodule'] = xray_result["nodule_density"]
        base_patient_profile['xray_pleural'] = xray_result["pleural_thickening"]
        
        # Add annotation to appear on the timeline exactly now
        active_annotations.append({
            "label": "X-Ray Analyzed",
            "message": f"Op:{xray_result['opacity_score']:.2f}, Cav:{xray_result['cavity_probability']:.2f}",
            "type": "info"
        })
        
        return {"status": "injected", "xray_analysis": xray_result}
    except Exception as e:
        return {"error": str(e)}

# ── KEEPING THE OLD ENDPOINTS FOR TAB 1 COMPATIBILITY ──

from pydantic import BaseModel
class PatientInput(BaseModel):
    vitals: dict

@app.post("/upload-xray")
async def upload_xray(file: UploadFile = File(...)):
    """Upload chest X-ray, extract features, run QML analysis (Static Tab)."""
    try:
        contents = await file.read()
        xray_result = analyze_xray_image(contents)
        
        # Run through QML with placeholder vitals
        placeholder_vitals = [78, 97.5, 16, 36.8, 7, 10, 2, 30, 13.5, 4.0,
                              xray_result["opacity_score"],
                              xray_result["cavity_probability"],
                              xray_result["nodule_density"],
                              xray_result["pleural_thickening"],
                              15, 5]
        
        qml_result = tb_engine.full_analysis(placeholder_vitals)
        
        return {
            "xray_analysis": xray_result,
            "qml_with_xray": qml_result,
            "advisory": xray_result.get("findings", []),
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze-patient")
async def analyze_patient(patient: PatientInput):
    """Static analysis endpoint for Tab 1"""
    features = [patient.vitals.get(name, 0.0) for name in FEATURE_NAMES]
    features_array = np.array(features)
    cl_prob = float(classical_rf.predict_proba([features_array])[0][1])
    cl_importances = classical_rf.feature_importances_.tolist()
    
    classical_features = []
    for i, name in enumerate(FEATURE_NAMES):
        classical_features.append({
            "name": name, "value": float(features_array[i]),
            "importance": float(cl_importances[i]), "status": "normal"
        })
    
    qml_result = tb_engine.full_analysis(features_array)
    return {
        "classical_ml": {"risk_score": cl_prob, "diagnosis": "Healthy" if cl_prob < 0.5 else "Anomaly", "features": classical_features},
        "quantum_ml": qml_result,
        "advisory": [] # Simplified
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
