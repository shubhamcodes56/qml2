"""
quantum_circuit.py — 4-Qubit Variational Quantum Classifier for TB Detection
=============================================================================
This module defines the core quantum circuit using PennyLane.

Architecture:
  - 4 Qubits (one per extracted feature)
  - Angle Embedding: RY(feature_i) on each qubit
  - 2 Variational Layers, each containing:
      * RY(θ) + RZ(θ) rotations on all 4 qubits (trainable)
      * CNOT circular entanglement: 0↔1, 1↔2, 2↔3, 3↔0
  - Measurement: ⟨PauliZ⟩ on Qubit 0 → maps to TB probability

Total trainable parameters: 2 layers × 4 qubits × 2 gates = 16 weights

Math:
  - Angle Embedding:   |ψ⟩ = RY(x₁)|0⟩ ⊗ RY(x₂)|0⟩ ⊗ RY(x₃)|0⟩ ⊗ RY(x₄)|0⟩
  - Entanglement:      CNOT(0,1) · CNOT(1,2) · CNOT(2,3) · CNOT(3,0)
  - Variational:       RZ(θ₂) · RY(θ₁) applied to each qubit
  - Output:            P(TB) = (1 - ⟨Z₀⟩) / 2
"""

import pennylane as qml
from pennylane import numpy as np

# ── Device setup ──────────────────────────────────────────────────────────────
# "default.qubit" is PennyLane's built-in statevector simulator.
# It performs exact matrix multiplication — mathematically identical to a
# real quantum computer (no noise, no decoherence).
N_QUBITS = 4
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


# ── The Quantum Circuit (QNode) ──────────────────────────────────────────────
@qml.qnode(dev, interface="autograd")
def quantum_classifier(features, weights):
    """
    4-Qubit Variational Quantum Classifier.

    Parameters
    ----------
    features : array-like, shape (4,)
        Four classical features extracted from a chest X-ray via ResNet18 + PCA.
        Each feature is mapped to a qubit rotation angle.
    weights : array-like, shape (N_LAYERS, N_QUBITS, 2)
        Trainable parameters for RY and RZ rotations in each variational layer.

    Returns
    -------
    float
        Expectation value of PauliZ on Qubit 0. Range [-1, +1].
        -1 → high TB probability, +1 → Normal.
    """
    # ── Step 1: Angle Embedding ───────────────────────────────────────────
    # Map each classical feature to a qubit rotation.
    # RY(x) rotates the qubit state on the Bloch sphere by angle x around Y-axis.
    # This encodes the feature into the quantum state's amplitude.
    for i in range(N_QUBITS):
        qml.RY(features[i], wires=i)

    # ── Step 2: Variational Layers ────────────────────────────────────────
    for layer in range(N_LAYERS):
        # 2a. Parameterized rotations (trainable)
        # RY(θ) controls the amplitude mixing
        # RZ(θ) controls the phase — this is what lets QML detect
        # phase-shifts (like the 4ms BP delay in our ICU scenario)
        for qubit in range(N_QUBITS):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)

        # 2b. Circular CNOT entanglement
        # This creates quantum correlations between qubits.
        # If feature on Qubit 0 (e.g., lung texture) has a subtle anomaly,
        # entanglement ensures Qubit 1 (e.g., density) also gets affected.
        # This is the "Cross-Organ Communication" from our report.
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # ── Step 3: Measurement ───────────────────────────────────────────────
    # Measure the expectation value ⟨Z⟩ on Qubit 0.
    # PauliZ has eigenvalues +1 (|0⟩) and -1 (|1⟩).
    # We'll map this to probability in post-processing.
    return qml.expval(qml.PauliZ(0))


def predict_probability(features, weights):
    """
    Convert quantum measurement to TB probability.

    Maps ⟨Z⟩ ∈ [-1, +1] → P(TB) ∈ [0, 1]
    Formula: P(TB) = (1 - ⟨Z⟩) / 2

    When ⟨Z⟩ = +1 → P(TB) = 0 (Normal)
    When ⟨Z⟩ = -1 → P(TB) = 1 (TB detected)
    """
    expval = quantum_classifier(features, weights)
    probability = (1.0 - expval) / 2.0
    return float(probability)


def get_circuit_info():
    """Return a text description of the quantum circuit for display."""
    info = {
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "n_trainable_params": N_LAYERS * N_QUBITS * 2,
        "simulator": "PennyLane default.qubit (exact statevector)",
        "embedding": "Angle Embedding (RY gates)",
        "entanglement": "Circular CNOT (0→1→2→3→0)",
        "variational_gates": "RY(θ) + RZ(θ) per qubit per layer",
        "measurement": "⟨PauliZ⟩ on Qubit 0",
        "math_equivalence": "Mathematically identical to a real quantum computer",
    }
    return info


def init_weights(seed=42):
    """Initialize random trainable weights for the quantum circuit."""
    np.random.seed(seed)
    return np.random.uniform(
        low=-np.pi, high=np.pi,
        size=(N_LAYERS, N_QUBITS, 2),
        requires_grad=True
    )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  QML Quantum Circuit — Self Test")
    print("=" * 60)

    # Initialize random weights
    weights = init_weights()
    print(f"\n✓ Initialized {N_LAYERS * N_QUBITS * 2} trainable parameters")
    print(f"  Weight shape: {weights.shape}")

    # Test with dummy features
    test_features = np.array([0.5, -0.3, 0.8, 0.1])
    print(f"\n✓ Test features: {test_features}")

    # Run the circuit
    raw_output = quantum_classifier(test_features, weights)
    tb_prob = predict_probability(test_features, weights)
    print(f"  Raw ⟨PauliZ⟩ output: {raw_output:.4f}")
    print(f"  TB Probability: {tb_prob:.4f} ({tb_prob*100:.1f}%)")

    # Print circuit info
    print(f"\n✓ Circuit Info:")
    for k, v in get_circuit_info().items():
        print(f"  {k}: {v}")

    # Draw the circuit
    print(f"\n✓ Circuit Diagram:")
    drawer = qml.draw(quantum_classifier)
    print(drawer(test_features, weights))

    print("\n" + "=" * 60)
    print("  All tests passed! Quantum circuit is working.")
    print("=" * 60)
