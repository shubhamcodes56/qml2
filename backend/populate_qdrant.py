import os
import torch
import numpy as np
import math
import pennylane as qml

from generate_tb_data import generate_tb_dataset
from qdrant_store import QuantumVectorStore

# Quantum Circuit setup (from compare_models.py)
N_QUBITS = 8
N_LAYERS = 4
dev = qml.device('default.qubit', wires=N_QUBITS)

NORM_BASES  = torch.tensor([78.0, 97.5, 16.0, 36.8, 7.0, 10.0, 2.0, 30.0, 13.5, 4.0, 0.08, 0.04, 0.06, 0.05, 15.0, 5.0])
NORM_SCALES = torch.tensor([16.0, -5.0, 8.0, 1.5, 6.0, 20.0, 8.0, 15.0, -4.0, -1.5, 0.30, 0.25, 0.25, 0.20, 40.0, 10.0])

def normalize_features(X_raw):
    return torch.clamp(((X_raw - NORM_BASES) / NORM_SCALES) * math.pi, -math.pi, math.pi)

@qml.qnode(dev, interface='torch')
def get_state_vector(inputs, weights):
    # Same circuit architecture
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
    
    return qml.state()

def populate_qdrant():
    print("1. Loading QML Weights...")
    try:
        qml_weights = torch.tensor(np.load('saved_model/tb_weights.npy'))
    except Exception as e:
        print("Error loading tb_weights.npy:", e)
        return

    print("2. Generating Historical Patients...")
    # Generate a small historical database of 500 patients
    X_hist, y_hist, t_hist = generate_tb_dataset(500)
    
    print("3. Connecting to Qdrant (Local Mode)...")
    store = QuantumVectorStore(persist_dir="qdrant_db")
    
    print("4. Extracting State Vectors and Storing...")
    patient_ids = []
    state_vectors = []
    metadata = []
    
    X_norm = normalize_features(torch.tensor(X_hist, dtype=torch.float32))
    
    for i in range(len(X_norm)):
        # Extract full 256-dim complex state vector from the quantum circuit!
        state_vec = get_state_vector(X_norm[i], qml_weights).detach().numpy()
        
        patient_ids.append(1000 + i)
        state_vectors.append(state_vec)
        
        label = "TB Positive" if y_hist[i] == 1 else "Healthy"
        severity = f"Tier-{t_hist[i]}" if t_hist[i] > 0 else "Normal"
        
        metadata.append({
            "diagnosis": label,
            "severity": severity,
            "tier": int(t_hist[i]),
            "risk_score": float(np.random.uniform(0.6, 0.99) if y_hist[i] == 1 else np.random.uniform(0.01, 0.3))
        })
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/500 patients...")

    # Bulk insert
    store.bulk_store(patient_ids, state_vectors, metadata, batch_size=100)
    print("Done! Qdrant is populated.")

if __name__ == "__main__":
    populate_qdrant()
