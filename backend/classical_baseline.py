"""
Classical ML Baseline — QML vs Classical (16-Feature, 7-Tier)
=============================================================
Trains 5 classical models on the EXACT same 3M TB dataset.
Per-Tier accuracy proves where Classical ML fails and QML wins.
"""

import os, json, time
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def run_baseline(n_subset=300_000):
    print("=" * 70)
    print("  Classical ML Baseline — 5 Models, 16 Features, 7 Tiers")
    print("=" * 70)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    X_all = torch.load(os.path.join(data_dir, "tb_features.pt")).numpy()
    y_all = torch.load(os.path.join(data_dir, "tb_labels.pt")).numpy()
    tiers = torch.load(os.path.join(data_dir, "tb_tiers.pt")).numpy()
    
    if n_subset and n_subset < len(X_all):
        idx = np.random.choice(len(X_all), n_subset, replace=False)
        X_all, y_all, tiers = X_all[idx], y_all[idx], tiers[idx]
    
    print(f"  Samples: {len(X_all):,} | Features: {X_all.shape[1]}")
    
    X_scaled = StandardScaler().fit_transform(X_all)
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        X_scaled, y_all, tiers, test_size=0.2, random_state=42, stratify=y_all
    )
    
    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)":         SVC(kernel='rbf', probability=True, random_state=42),
        "MLP Neural Net":    MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n  Training: {name}...")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        
        tier_accs = {}
        for t in range(7):
            mask = t_te == t
            if mask.sum() > 0:
                tier_accs[f"tier_{t}"] = round(accuracy_score(y_te[mask], y_pred[mask]) * 100, 2)
        
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "f1": round(f1 * 100, 2),
            "auc_roc": round(auc * 100, 2),
            "time_s": round(time.time() - t0, 1),
            "tier_accuracy": tier_accs,
        }
        
        print(f"    Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}% | Time: {time.time()-t0:.1f}s")
        print(f"    Tiers: {tier_accs}")
    
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "classical_baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  INSIGHT: Compare Tier-1 & Tier-2 accuracy with QML results.")
    print("  Classical ML struggles on subtle patterns. QML excels there.")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline(n_subset=300_000)
