import os
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

# 4 Qubits for 4 Vitals
N_QUBITS = 4
N_LAYERS = 2

# Using default.qubit simulator which supports full state vector access
dev = qml.device("default.qubit", wires=N_QUBITS)

def normalize_vitals(vitals_dict):
    """
    Normalize vitals into [-pi, pi] range for quantum angle embedding.
    """
    baselines = {
        "Heart Rate": (75.0, 30.0),
        "SpO2": (98.0, -10.0),
        "Resp Rate": (16.0, 10.0),
        "Temperature": (37.0, 2.0)
    }
    
    normalized = []
    keys = ["Heart Rate", "SpO2", "Resp Rate", "Temperature"]
    
    for key in keys:
        val = vitals_dict[key]
        base, scale = baselines[key]
        norm_val = (val - base) / scale
        angle = norm_val * np.pi
        angle = np.clip(angle, -np.pi, np.pi)
        normalized.append(angle)
        
    return pnp.array(normalized, requires_grad=False)

@qml.qnode(dev)
def qml_state_circuit(features, weights):
    """
    Returns the FULL quantum state vector (16 complex amplitudes)
    instead of just a single expectation value.
    """
    # 1. State Preparation: Angle Embedding
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

    # 2. Variational Entanglement Layers
    for layer in range(N_LAYERS):
        # Entanglement
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
            
        # Variational Rotations
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

    # Return the full state vector
    return qml.state()


class QMLVitalsEngine:
    def __init__(self):
        # We try to load the dynamically trained weights from the PyTorch pipeline
        model_path = os.path.join(os.path.dirname(__file__), "saved_model", "qml_ts_weights.npy")
        
        if os.path.exists(model_path):
            print("[QML ENGINE] Loaded Advanced PyTorch-Trained Weights!")
            weights_np = np.load(model_path)
            self.weights = pnp.array(weights_np, requires_grad=False)
        else:
            print("[QML ENGINE] Using hardcoded weights (Run train_ts.py for advanced ML)")
            # Fallback for demonstration if training hasn't been run
            self.weights = pnp.array([
                [[ 1.2, -0.5], [ 0.8,  1.1], [-1.5,  0.2], [ 0.4, -0.9]],
                [[-0.8,  1.4], [-1.1, -0.3], [ 0.9,  0.7], [-0.5,  1.6]]
            ], requires_grad=False)
        
    def analyze_vitals(self, vitals_dict):
        """
        Analyze vitals and return the deep mathematical QML state.
        """
        features = normalize_vitals(vitals_dict)
        
        # Get full 16-dimensional complex state vector
        state_vector = qml_state_circuit(features, self.weights)
        
        # Convert state vector to probabilities (Amplitude squared)
        probabilities = np.abs(state_vector)**2
        
        # Calculate Density Matrix (Outer product of state vector with its conjugate transpose)
        # This is a 16x16 matrix where off-diagonal elements show phase coherence/entanglement
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        
        # We'll extract just the magnitude of the density matrix for visualization
        density_matrix_abs = np.abs(density_matrix)
        
        # Calculate a single "Risk Score" based on the probability of the state collapsing 
        # into the |1111> basis state (index 15), representing total system failure.
        # In a real model, this would be a weighted sum or a specific measurement projection.
        anomaly_prob = float(np.sum(probabilities[8:])) # Probability of Qubit 0 being |1>
        
        # Exaggerate curve for UI
        squashed_prob = 1 / (1 + np.exp(-10 * (anomaly_prob - 0.3)))
        risk_score = min(100, max(0, int(squashed_prob * 100)))
        
        severity = "NORMAL"
        if risk_score > 85:
            severity = "CRITICAL"
        elif risk_score > 60:
            severity = "WARNING"
        elif risk_score > 30:
            severity = "LOW"
            
        # Format the state vector for JSON serialization (complex numbers aren't JSON serializable)
        formatted_state = [{"real": float(c.real), "imag": float(c.imag)} for c in state_vector]
            
        return {
            "risk_score": risk_score,
            "severity": severity,
            "probabilities": probabilities.tolist(),
            "density_matrix_heatmap": density_matrix_abs.tolist(), # 16x16 2D array
            "state_vector": formatted_state,
            "circuit_angles": [float(f) for f in features]
        }
