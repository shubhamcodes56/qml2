import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

# ── 1. Advanced Architecture Setup ──────────────────────────────────────────
N_QUBITS = 4
N_LAYERS = 2

# For heavy training, we use default.qubit. 
# On Kaggle, you can change this to 'lightning.qubit' or 'default.qubit.torch' for GPU acceleration.
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    """
    The core mathematical model. 
    inputs: The classical vitals mapped to angles [-pi, pi]
    weights: The trainable entanglement and rotation parameters
    """
    # 1. Non-linear mapping to Hilbert Space (Supports automatic batching)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')

    # 2. Multivariate Entanglement Learning
    for layer in range(N_LAYERS):
        # Learn topological correlations
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        
        # Variational tuning
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

    # 3. Measurement (Expectation value of Pauli-Z on Qubit 0)
    return qml.expval(qml.PauliZ(0))

class QMLPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Random initial weights
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 2)}
        init_weights = torch.randn(weight_shapes["weights"]) * 0.1
        
        # Turn PennyLane circuit into PyTorch neural network layer
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=lambda w: init_weights)

    def forward(self, x):
        # The QNode returns [-1, 1]. We scale it slightly so BCEWithLogitsLoss works well.
        return self.qlayer(x).squeeze()


# ── 2. Data Loading & Normalization ───────────────────────────────────────
def normalize_tensor(vitals_tensor):
    """Normalize raw vitals into quantum angles [-pi, pi]"""
    # Baselines: HR=75, SpO2=98, RR=16, Temp=37
    # Scales: 30, -10, 10, 2
    # Tensor shape: (Batch, 4)
    
    baselines = torch.tensor([75.0, 98.0, 16.0, 37.0], device=vitals_tensor.device)
    scales = torch.tensor([30.0, -10.0, 10.0, 2.0], device=vitals_tensor.device)
    
    norm_val = (vitals_tensor - baselines) / scales
    angles = norm_val * np.pi
    return torch.clamp(angles, -np.pi, np.pi)


# ── 3. The Core Training Loop ─────────────────────────────────────────────
def train_model(epochs=15, batch_size=32, lr=0.01):
    print("=" * 60)
    print("  Advanced QML Time-Series Training Pipeline (PyTorch)")
    print("=" * 60)
    
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(os.path.join(data_dir, "icu_features.pt")):
        print("Error: Run generate_icu_data.py first to generate the dataset!")
        return
        
    X_raw = torch.load(os.path.join(data_dir, "icu_features.pt"))
    y_raw = torch.load(os.path.join(data_dir, "icu_labels.pt"))
    
    # Normalize features for quantum embedding
    X = normalize_tensor(X_raw)
    
    # Split Train/Test (80/20)
    dataset_size = len(X)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        TensorDataset(X, y_raw), [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Loaded {dataset_size} sequences (Train: {train_size}, Test: {test_size})")
    
    # 2. Initialize Model
    model = QMLPredictor()
    
    # Loss and Optimizer
    # We use BCEWithLogitsLoss because QNode returns continuous values.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting Training Engine...")
    print(f"Model Parameters: {N_LAYERS * N_QUBITS * 2}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            
            # Loss computation
            loss = criterion(outputs, batch_y)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Accuracy (sigmoid > 0.5 means class 1)
            predictions = torch.sigmoid(outputs) >= 0.5
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        # Save best model weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the raw numpy weights for the quantum_vitals.py engine
            trained_weights = model.qlayer.weights.detach().numpy()
            model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
            os.makedirs(model_dir, exist_ok=True)
            np.save(os.path.join(model_dir, "qml_ts_weights.npy"), trained_weights)
            
    print("\n============================================================")
    print("  Training Complete!")
    print(f"  Best weights saved to saved_model/qml_ts_weights.npy")
    print("============================================================")

if __name__ == "__main__":
    # We run with 15 epochs and batch_size 32 for quick convergence demonstration.
    # On Kaggle, you can increase epochs=50 and n_samples=10,000 for maximum accuracy.
    train_model(epochs=15, batch_size=64, lr=0.05)
