"""
Training Pipeline — 8-Qubit, 16-Feature TB Model
=================================================
Designed to run on Kaggle Free Tier GPU.
Manual batching for 16→8 qubit data re-uploading.
"""

import os, sys, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import numpy as np

N_QUBITS = 8
N_FEATURES = 16
N_LAYERS = 4

dev = qml.device("default.qubit", wires=N_QUBITS)

# Normalization constants (must match quantum_tb_engine.py)
NORM_BASES  = torch.tensor([78.0, 97.5, 16.0, 36.8, 7.0, 10.0, 2.0, 30.0, 13.5, 4.0, 0.08, 0.04, 0.06, 0.05, 15.0, 5.0])
NORM_SCALES = torch.tensor([16.0, -5.0, 8.0, 1.5, 6.0, 20.0, 8.0, 15.0, -4.0, -1.5, 0.30, 0.25, 0.25, 0.20, 40.0, 10.0])

def normalize_batch(X):
    return torch.clamp(((X - NORM_BASES) / NORM_SCALES) * np.pi, -np.pi, np.pi)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def train_circuit(inputs, weights):
    """Single-sample circuit matching quantum_tb_engine.py architecture."""
    for layer in range(N_LAYERS):
        if layer == 0:
            for q in range(N_QUBITS):
                qml.RY(inputs[q], wires=q)
        elif layer == 1:
            for q in range(N_QUBITS):
                qml.RY(inputs[q + 8], wires=q)
        elif layer == 2:
            for q in range(N_QUBITS):
                qml.RY(inputs[q], wires=q)
                qml.RZ(inputs[q + 8], wires=q)
        else:
            cross = [0, 10, 1, 11, 2, 12, 3, 13]
            for q in range(N_QUBITS):
                qml.RY(inputs[cross[q]], wires=q)
            for q in range(4):
                qml.RZ(inputs[14], wires=q)
                qml.RZ(inputs[15], wires=q + 4)
        
        for q in range(N_QUBITS):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
        
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        for i in range(0, N_QUBITS - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(4):
            qml.CNOT(wires=[i, i + 4])
    
    H = sum(qml.PauliZ(i) for i in range(N_QUBITS))
    return qml.expval(H)


class QMLTBPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 3) * 0.3)
    
    def forward(self, x):
        return torch.stack([train_circuit(x[i], self.weights) for i in range(x.shape[0])])
    
    def get_quantum_weights(self):
        return self.weights.detach().cpu().numpy()


def train_model(epochs=20, batch_size=32, lr=0.01, train_subset=10_000, patience=5):
    print("=" * 70)
    print("  QML TB Training — 8 Qubits | 16 Features | 256-dim Hilbert Space")
    print("=" * 70)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(os.path.join(data_dir, "tb_features.pt")):
        print("ERROR: Run generate_tb_data.py first!"); sys.exit(1)
    
    X_raw = torch.load(os.path.join(data_dir, "tb_features.pt"))
    y_all = torch.load(os.path.join(data_dir, "tb_labels.pt"))
    print(f"  Full dataset: {len(X_raw):,}")
    
    if train_subset and train_subset < len(X_raw):
        idx = torch.randperm(len(X_raw))[:train_subset]
        X_raw, y_all = X_raw[idx], y_all[idx]
        print(f"  Subset: {train_subset:,}")
    
    X = normalize_batch(X_raw)
    n_val = int(len(X) * 0.15)
    n_train = len(X) - n_val
    
    ds = TensorDataset(X, y_all)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Batch: {batch_size}")
    
    model = QMLTBPredictor()
    print(f"  Quantum params: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    no_improve = 0
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'Time':>6}")
    print("-" * 55)
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            tl += loss.item() * bx.size(0)
            tc += ((torch.sigmoid(out) >= 0.5).float() == by).sum().item()
            tt += bx.size(0)
        tl /= tt; ta = 100.0 * tc / tt
        
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                vl += loss.item() * bx.size(0)
                vc += ((torch.sigmoid(out) >= 0.5).float() == by).sum().item()
                vt += bx.size(0)
        vl /= max(vt, 1); va = 100.0 * vc / max(vt, 1)
        
        scheduler.step()
        print(f"{epoch+1:>4} | {tl:>8.4f} | {ta:>6.2f}% | {vl:>8.4f} | {va:>6.2f}% | {time.time()-t0:>5.1f}s")
        
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta); history["val_acc"].append(va)
        
        if vl < best_val_loss:
            best_val_loss = vl; no_improve = 0
            np.save(os.path.join(save_dir, "tb_weights.npy"), model.get_quantum_weights())
            torch.save(model.state_dict(), os.path.join(save_dir, "tb_full_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}"); break
    
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n  Done! Best Val Loss: {best_val_loss:.4f}")
    print(f"  Weights: saved_model/tb_weights.npy")


if __name__ == "__main__":
    train_model(epochs=20, batch_size=32, lr=0.01, train_subset=10_000)
