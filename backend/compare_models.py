import os
import torch
import numpy as np
import math
import pennylane as qml
from sklearn.ensemble import RandomForestClassifier

# Import your dynamic dataset generator
from generate_tb_data import generate_tb_dataset

# ==========================================
# 1. Quantum Model Setup
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
# 2. Dynamic Comparison Engine
# ==========================================
def run_comparison():
    print("1. Generating 10,000 new random patients dynamically...")
    X_all, y_all, t_all = generate_tb_dataset(10000)
    
    print("2. Training Classical Model (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_all, y_all)
    print("   [OK] Classical Model Ready.")
    
    print("3. Loading Quantum ML Model...")
    try:
        qml_weights = torch.tensor(np.load('saved_model/tb_weights.npy'))
        print("   [OK] Quantum Model Loaded.")
    except Exception as e:
        print("   Error loading tb_weights.npy:", e)
        return

    print("\n" + "="*60)
    print("   DYNAMIC PATIENT COMPARISON (CLASSICAL vs QUANTUM)")
    print("="*60)
    
    # Select 3 specific patients dynamically from the generated set
    test_cases = []
    
    # Healthy (Tier-0)
    idx_t0 = np.where(t_all == 0)[0][5]  # Pick 5th healthy person
    test_cases.append(("Healthy Person (Tier-0)", X_all[idx_t0], y_all[idx_t0]))
    
    # Early TB (Tier-1) - The tricky one
    idx_t1 = np.where(t_all == 1)[0][10] # Pick 10th Tier-1 person
    test_cases.append(("Hidden/Early TB (Tier-1)", X_all[idx_t1], y_all[idx_t1]))
    
    # Severe TB (Tier-6)
    idx_t6 = np.where(t_all == 6)[0][2]  # Pick 2nd Tier-6 person
    test_cases.append(("Severe TB (Tier-6)", X_all[idx_t6], y_all[idx_t6]))
    
    for name, features, true_label in test_cases:
        # Classical Prediction
        prob_cl = rf.predict_proba([features])[0][1]
        pred_cl = "TB Detected" if prob_cl >= 0.5 else "Healthy"
        
        # Quantum Prediction
        X_norm = normalize_features(torch.tensor([features], dtype=torch.float32))
        raw_out = quantum_circuit(X_norm[0], qml_weights)
        prob_qml = torch.sigmoid(raw_out).item()
        pred_qml = "TB Detected" if prob_qml >= 0.5 else "Healthy"
        
        print(f"\nPatient Profile: {name}")
        print(f"  Actual Status   : {'TB Positive' if true_label == 1 else 'Healthy'}")
        
        # Determine correctness
        cl_correct = "[OK]" if (prob_cl >= 0.5) == true_label else "[FAILED]"
        qm_correct = "[OK]" if (prob_qml >= 0.5) == true_label else "[FAILED]"
        
        print(f"  Classical ML    : {prob_cl*100:5.1f}% Risk -> [{pred_cl}] {cl_correct}")
        print(f"  Quantum ML      : {prob_qml*100:5.1f}% Risk -> [{pred_qml}] {qm_correct}")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    run_comparison()
