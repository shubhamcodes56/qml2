import torch
import numpy as np
import math
import pennylane as qml

# ==========================================
# 1. Quantum Model Setup (Same as Kaggle)
# ==========================================
N_QUBITS = 8
N_LAYERS = 4
dev = qml.device('default.qubit', wires=N_QUBITS)

NORM_BASES  = torch.tensor([78.0, 97.5, 16.0, 36.8, 7.0, 10.0, 2.0, 30.0, 13.5, 4.0, 0.08, 0.04, 0.06, 0.05, 15.0, 5.0])
NORM_SCALES = torch.tensor([16.0, -5.0, 8.0, 1.5, 6.0, 20.0, 8.0, 15.0, -4.0, -1.5, 0.30, 0.25, 0.25, 0.20, 40.0, 10.0])

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
    
    return qml.expval(sum(qml.PauliZ(i) for i in range(N_QUBITS)))

# ==========================================
# 2. Convert Single Patient Data to Row
# ==========================================
def predict_patient(patient_data_dict, weights_path='saved_model/tb_weights.npy'):
    # Load weights downloaded from Kaggle
    try:
        trained_weights = torch.tensor(np.load(weights_path))
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Please download it from Kaggle and place it in the saved_model folder.")
        return

    # Map the dictionary to a single 1D array in the exact order of features
    FEATURE_NAMES = [
        'heart_rate', 'spo2', 'resp_rate', 'temperature',
        'wbc_count', 'esr', 'crp', 'lymphocyte_pct',
        'hemoglobin', 'albumin',
        'xray_opacity', 'xray_cavity', 'xray_nodule', 'xray_pleural',
        'ada_level', 'mantoux_mm'
    ]
    
    # Create a row of shape (1, 16)
    row_data = [patient_data_dict.get(f, 0.0) for f in FEATURE_NAMES]
    X_raw = torch.tensor([row_data], dtype=torch.float32)
    
    # Normalize
    X_norm = normalize_features(X_raw)
    
    # Predict (Forward pass)
    with torch.no_grad():
        raw_output = quantum_circuit(X_norm[0], trained_weights)
        probability = torch.sigmoid(raw_output).item()
        
    print("-" * 40)
    print("PATIENT PREDICTION RESULT")
    print("-" * 40)
    print(f"Quantum Entanglement Score (Raw): {raw_output.item():.4f}")
    print(f"TB Risk Probability:              {probability * 100:.2f}%")
    if probability >= 0.5:
        print("Diagnosis:                        [ALERT] HIGH RISK (TB Detected)")
    else:
        print("Diagnosis:                        [OK] LOW RISK (Healthy)")
    print("-" * 40)

# ==========================================
# 3. Example Usage
# ==========================================
if __name__ == "__main__":
    # Example: A patient with mild (Tier-1) hidden symptoms
    sample_patient = {
        'heart_rate': 85.0, 
        'spo2': 96.0, 
        'resp_rate': 18.0, 
        'temperature': 37.2,
        'wbc_count': 8.5, 
        'esr': 15.0, 
        'crp': 3.0, 
        'lymphocyte_pct': 28.0,
        'hemoglobin': 12.5, 
        'albumin': 3.8,
        'xray_opacity': 0.15, 
        'xray_cavity': 0.05, 
        'xray_nodule': 0.08, 
        'xray_pleural': 0.06,
        'ada_level': 20.0, 
        'mantoux_mm': 8.0
    }
    
    print("Converting input data to a single row format (1x16 tensor)...")
    predict_patient(sample_patient)
