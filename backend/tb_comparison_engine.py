import torch
import numpy as np
import math
import pennylane as qml
from sklearn.ensemble import RandomForestClassifier

from generate_tb_data import generate_tb_dataset
from qdrant_store import QuantumVectorStore

# ==========================================
# 1. Quantum Model Setup
# ==========================================
N_QUBITS = 8
N_LAYERS = 4
dev = qml.device('default.qubit', wires=N_QUBITS)

NORM_BASES  = torch.tensor([78.0, 97.5, 16.0, 36.8, 7.0, 10.0, 2.0, 30.0, 13.5, 4.0, 0.08, 0.04, 0.06, 0.05, 15.0, 5.0])
NORM_SCALES = torch.tensor([16.0, -5.0, 8.0, 1.5, 6.0, 20.0, 8.0, 15.0, -4.0, -1.5, 0.30, 0.25, 0.25, 0.20, 40.0, 10.0])

FEATURE_NAMES = [
    'heart_rate', 'spo2', 'resp_rate', 'temperature',
    'wbc_count', 'esr', 'crp', 'lymphocyte_pct',
    'hemoglobin', 'albumin',
    'xray_opacity', 'xray_cavity', 'xray_nodule', 'xray_pleural',
    'ada_level', 'mantoux_mm'
]

def normalize_features(X_raw):
    return torch.clamp(((X_raw - NORM_BASES) / NORM_SCALES) * math.pi, -math.pi, math.pi)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    for q in range(N_QUBITS): qml.RY(inputs[q], wires=q)
    for q in range(N_QUBITS): qml.Rot(weights[0,q,0], weights[0,q,1], weights[0,q,2], wires=q)
    for i in range(N_QUBITS): qml.CNOT(wires=[i,(i+1)%N_QUBITS])
    
    for q in range(N_QUBITS): qml.RY(inputs[q+8], wires=q)
    for q in range(N_QUBITS): qml.Rot(weights[1,q,0], weights[1,q,1], weights[1,q,2], wires=q)
    for i in range(N_QUBITS): qml.CNOT(wires=[i,(i+1)%N_QUBITS])
    
    for q in range(N_QUBITS):
        qml.RY(inputs[q], wires=q)
        qml.RZ(inputs[q+8], wires=q)
    for q in range(N_QUBITS): qml.Rot(weights[2,q,0], weights[2,q,1], weights[2,q,2], wires=q)
    for i in range(N_QUBITS): qml.CNOT(wires=[i,(i+1)%N_QUBITS])
    
    cross = [0, 10, 1, 11, 2, 12, 3, 13]
    for q in range(N_QUBITS): qml.RY(inputs[cross[q]], wires=q)
    for q in range(4):
        qml.RZ(inputs[14], wires=q)
        qml.RZ(inputs[15], wires=q+4)
    for q in range(N_QUBITS): qml.Rot(weights[3,q,0], weights[3,q,1], weights[3,q,2], wires=q)
    for i in range(N_QUBITS): qml.CNOT(wires=[i,(i+1)%N_QUBITS])
    
    return qml.expval(sum(qml.PauliZ(i) for i in range(N_QUBITS))), qml.state()

# ==========================================
# 2. TB Comparison Engine Class
# ==========================================
class TBComparisonEngine:
    def __init__(self):
        print("[TB-Engine] Generating Classical Training Dataset (1000 patients)...")
        self.X_all, self.y_all, self.t_all = generate_tb_dataset(1000)
        
        print("[TB-Engine] Training Classical Random Forest...")
        self.rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.rf.fit(self.X_all, self.y_all)
        
        print("[TB-Engine] Loading Quantum Model...")
        self.qml_weights = torch.tensor(np.load('saved_model/tb_weights.npy'))
        
        print("[TB-Engine] Connecting to Qdrant Local...")
        try:
            self.qdrant = QuantumVectorStore(persist_dir="qdrant_db")
        except Exception as e:
            print("[TB-Engine] Qdrant Error:", e)
            self.qdrant = None
            
        print("[TB-Engine] Ready.")

    def generate_and_compare(self):
        # Pick a random Tier-1 patient
        idx_t1 = np.random.choice(np.where(self.t_all == 1)[0])
        features = self.X_all[idx_t1]
        
        # Format for UI
        patient_data = {FEATURE_NAMES[i]: float(features[i]) for i in range(16)}
        
        # Classical Prediction
        prob_cl = self.rf.predict_proba([features])[0][1]
        
        # Quantum Prediction
        X_norm = normalize_features(torch.tensor([features], dtype=torch.float32))
        raw_out, state_vec = quantum_circuit(X_norm[0], self.qml_weights)
        prob_qml = torch.sigmoid(raw_out).item()
        
        # Similar patients via Qdrant
        similar_patients = []
        if self.qdrant:
            try:
                # state_vec is a tensor containing complex numbers, we detach it to numpy
                sv_np = state_vec.detach().numpy()
                similar_patients = self.qdrant.find_similar(sv_np, top_k=3)
            except Exception as e:
                print("Error finding similar patients:", e)
        
        return {
            "patient_data": patient_data,
            "actual_tier": "Tier-1 (Hidden TB)",
            "classical_ml": {
                "risk_score": float(prob_cl),
                "diagnosis": "TB Detected" if prob_cl >= 0.5 else "Healthy (Missed)"
            },
            "quantum_ml": {
                "risk_score": float(prob_qml),
                "diagnosis": "TB Detected" if prob_qml >= 0.5 else "Healthy (Missed)"
            },
            "qdrant_similar": similar_patients
        }
