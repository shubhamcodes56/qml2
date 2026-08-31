"""
train.py — Train the 4-Qubit Variational Quantum Classifier
=============================================================
This script trains the quantum circuit on synthetic TB vs Normal data
using PennyLane's gradient descent with the parameter-shift rule.

Training Process:
  1. Load synthetic features (4D) and labels (0=Normal, 1=TB)
  2. Initialize random quantum circuit weights
  3. For each epoch:
     - Forward pass: features → quantum circuit → prediction
     - Compute binary cross-entropy loss
     - Backward pass: parameter-shift rule computes quantum gradients
     - Update weights with Adam optimizer
  4. Save trained weights to saved_model/qml_weights.npy

The Parameter-Shift Rule:
  Unlike classical backpropagation, quantum gradients are computed by
  shifting each parameter by ±π/2 and measuring the difference:
    ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
  This is a real quantum gradient method used on actual quantum hardware.
"""

import os
import sys
import time
import numpy as np
import pennylane as qml

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from quantum_circuit import quantum_classifier, init_weights, N_QUBITS, N_LAYERS
from pennylane import numpy as pnp


def binary_cross_entropy(prediction, label):
    """
    Binary cross-entropy loss.
    prediction: raw ⟨Z⟩ ∈ [-1, +1], mapped to probability
    label: 0 (Normal) or 1 (TB)
    """
    # Map ⟨Z⟩ to probability: P = (1 - ⟨Z⟩) / 2
    prob = (1.0 - prediction) / 2.0
    # Clip for numerical stability
    prob = pnp.clip(prob, 1e-7, 1 - 1e-7)
    # BCE loss — use plain float for label to avoid autograd issues
    lab = float(label)
    loss = -(lab * pnp.log(prob) + (1.0 - lab) * pnp.log(1.0 - prob))
    return loss


def cost_function(weights, features_batch, labels_batch):
    """
    Average loss over a batch of samples.

    Parameters
    ----------
    weights : array, shape (N_LAYERS, N_QUBITS, 2)
        Trainable quantum circuit parameters.
    features_batch : array, shape (batch_size, 4)
        Batch of feature vectors.
    labels_batch : array, shape (batch_size,)
        Batch of labels (0 or 1).

    Returns
    -------
    float
        Mean binary cross-entropy loss over the batch.
    """
    total_loss = 0.0
    for features, label in zip(features_batch, labels_batch):
        prediction = quantum_classifier(features, weights)
        total_loss += binary_cross_entropy(prediction, label)
    return total_loss / len(labels_batch)


def compute_accuracy(weights, features, labels):
    """Compute classification accuracy."""
    correct = 0
    for feat, label in zip(features, labels):
        raw = float(quantum_classifier(pnp.array(feat, requires_grad=False), weights))
        prob = (1.0 - raw) / 2.0
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            correct += 1
    return correct / len(labels)


def train(
    n_epochs=25,
    batch_size=20,
    learning_rate=0.05,
    data_dir=None,
    save_dir=None,
):
    """
    Train the quantum classifier.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Number of samples per gradient update.
    learning_rate : float
        Adam optimizer learning rate.
    data_dir : str
        Path to directory containing features.npy and labels.npy.
    save_dir : str
        Path to save trained weights.
    """
    print("=" * 60)
    print("  QML Training — 4-Qubit Variational Classifier")
    print("=" * 60)

    # ── Load Data ─────────────────────────────────────────────────────────
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "saved_model")

    os.makedirs(save_dir, exist_ok=True)

    features = np.load(os.path.join(data_dir, "features.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    print(f"\n✓ Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"  TB: {int(labels.sum())}, Normal: {int(len(labels) - labels.sum())}")

    # ── Train/Test Split (80/20) ──────────────────────────────────────────
    n_train = int(0.8 * len(labels))
    train_features = pnp.array(features[:n_train], requires_grad=False)
    train_labels = pnp.array(labels[:n_train], requires_grad=False)
    test_features = pnp.array(features[n_train:], requires_grad=False)
    test_labels = pnp.array(labels[n_train:], requires_grad=False)

    print(f"  Train: {n_train}, Test: {len(labels) - n_train}")

    # ── Initialize Weights ────────────────────────────────────────────────
    weights = init_weights(seed=42)
    print(f"\n✓ Initialized {weights.size} trainable parameters")
    print(f"  Shape: {weights.shape} (layers × qubits × gates)")

    # ── Optimizer ─────────────────────────────────────────────────────
    # PennyLane's Adam optimizer supports quantum parameter-shift gradients
    opt = qml.optimize.AdamOptimizer(stepsize=learning_rate)

    print(f"\n✓ Optimizer: Adam (lr={learning_rate})")
    print(f"  Gradient method: Parameter-Shift Rule")
    print(f"  Epochs: {n_epochs}, Batch size: {batch_size}")
    print(f"\n{'='*60}")
    print(f"  {'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Time':<10}")
    print(f"{'='*60}")

    # ── Training Loop ─────────────────────────────────────────────────────
    best_acc = 0.0
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, n_epochs + 1):
        t_start = time.time()

        # Shuffle training data each epoch
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i: i + batch_size]
            batch_features = pnp.array(features[batch_idx], requires_grad=False)
            batch_labels = pnp.array(labels[batch_idx], requires_grad=False)

            # Compute loss and gradients (parameter-shift rule)
            def cost_fn(w):
                return cost_function(w, batch_features, batch_labels)

            weights, loss_val = opt.step_and_cost(cost_fn, weights)
            epoch_loss += float(loss_val)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t_start

        # Compute accuracy every 5 epochs (expensive for quantum circuits)
        if epoch % 5 == 0 or epoch == 1:
            # Use subset for speed
            subset_size = min(50, n_train)
            train_acc = compute_accuracy(
                weights,
                features[:subset_size],
                labels[:subset_size]
            )
            history["train_acc"].append(train_acc)
            print(f"  {epoch:<8} {avg_loss:<12.4f} {train_acc*100:<12.1f}% {elapsed:<10.1f}s")

            if train_acc > best_acc:
                best_acc = train_acc
                # Save best weights
                np.save(os.path.join(save_dir, "qml_weights.npy"), np.array(weights))
        else:
            print(f"  {epoch:<8} {avg_loss:<12.4f} {'...':<12} {elapsed:<10.1f}s")

        history["loss"].append(avg_loss)

    # ── Final Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final Evaluation")
    print(f"{'='*60}")

    # Load best weights
    best_weights = pnp.array(
        np.load(os.path.join(save_dir, "qml_weights.npy")),
        requires_grad=True
    )

    test_acc = compute_accuracy(best_weights, test_features, test_labels)
    train_acc_final = compute_accuracy(
        best_weights,
        train_features[:50],
        train_labels[:50]
    )

    print(f"  Train Accuracy: {train_acc_final*100:.1f}%")
    print(f"  Test Accuracy:  {test_acc*100:.1f}%")
    print(f"  Best weights saved to: {save_dir}/qml_weights.npy")

    # Save training history
    import json
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"{'='*60}")

    return best_weights, history


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train(
        n_epochs=15,
        batch_size=5,
        learning_rate=0.1,
    )

