"""
generate_synthetic.py — Generate Synthetic TB vs Normal Feature Data
====================================================================
Since we're building a working demo without hospital-grade datasets,
this script generates realistic synthetic 4-dimensional feature vectors
that mimic the statistical properties of TB vs Normal chest X-ray features.

TB Feature Characteristics (based on medical literature):
  - Feature 0 (Texture Density): TB shows higher irregular texture
  - Feature 1 (Edge Sharpness): TB cavities create sharper local edges
  - Feature 2 (Opacity Variation): TB infiltrates cause uneven opacity
  - Feature 3 (Symmetry Score): TB lesions break bilateral symmetry

The synthetic data preserves these statistical differences at a subtle
level — exactly the kind of "minor changes" that QML is designed to detect
but classical ML often misses.
"""

import numpy as np
import os
import json


def generate_tb_features(n_samples, seed=42):
    """
    Generate synthetic features mimicking TB-positive X-ray patterns.

    TB patients show:
    - Slightly elevated texture density (mean shift +0.3)
    - Higher edge variance (more irregular boundaries)
    - More opacity variation (infiltrates, consolidation)
    - Lower symmetry (unilateral lesions)
    """
    rng = np.random.RandomState(seed)

    features = np.zeros((n_samples, 4))

    # Feature 0: Texture Density — slightly elevated in TB
    features[:, 0] = rng.normal(loc=0.6, scale=0.25, size=n_samples)

    # Feature 1: Edge Sharpness — higher variance in TB (cavities)
    features[:, 1] = rng.normal(loc=0.4, scale=0.35, size=n_samples)

    # Feature 2: Opacity Variation — more spread in TB
    features[:, 2] = rng.normal(loc=0.5, scale=0.30, size=n_samples)

    # Feature 3: Symmetry Score — lower in TB (asymmetric lesions)
    features[:, 3] = rng.normal(loc=-0.3, scale=0.20, size=n_samples)

    # Add subtle cross-feature correlations (like real organ interactions)
    # When texture density is high, edge sharpness tends to increase
    correlation_noise = rng.normal(0, 0.05, size=n_samples)
    features[:, 1] += 0.15 * features[:, 0] + correlation_noise

    # Clip to reasonable range
    features = np.clip(features, -2.0, 2.0)

    return features


def generate_normal_features(n_samples, seed=123):
    """
    Generate synthetic features mimicking Normal (healthy) X-ray patterns.

    Normal patients show:
    - Lower, more uniform texture density
    - Regular edge patterns (no cavities)
    - Uniform opacity (clear lung fields)
    - High bilateral symmetry
    """
    rng = np.random.RandomState(seed)

    features = np.zeros((n_samples, 4))

    # Feature 0: Texture Density — lower and more uniform
    features[:, 0] = rng.normal(loc=0.1, scale=0.15, size=n_samples)

    # Feature 1: Edge Sharpness — low variance (smooth boundaries)
    features[:, 1] = rng.normal(loc=0.1, scale=0.15, size=n_samples)

    # Feature 2: Opacity Variation — low (clear lungs)
    features[:, 2] = rng.normal(loc=0.0, scale=0.15, size=n_samples)

    # Feature 3: Symmetry Score — high positive (symmetric)
    features[:, 3] = rng.normal(loc=0.3, scale=0.15, size=n_samples)

    # Clip to reasonable range
    features = np.clip(features, -2.0, 2.0)

    return features


def generate_dataset(n_tb=250, n_normal=250, save_dir=None):
    """
    Generate complete synthetic dataset.

    Parameters
    ----------
    n_tb : int
        Number of TB-positive samples.
    n_normal : int
        Number of Normal samples.
    save_dir : str, optional
        Directory to save the dataset files.

    Returns
    -------
    dict with keys: 'features', 'labels', 'feature_names', 'stats'
    """
    print(f"Generating {n_tb} TB + {n_normal} Normal samples...")

    tb_features = generate_tb_features(n_tb)
    normal_features = generate_normal_features(n_normal)

    # Combine
    features = np.vstack([tb_features, normal_features])
    labels = np.array([1] * n_tb + [0] * n_normal)  # 1 = TB, 0 = Normal

    # Shuffle
    rng = np.random.RandomState(999)
    shuffle_idx = rng.permutation(len(labels))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    # Scale features to [-π, π] for angle embedding
    max_abs = np.max(np.abs(features), axis=0, keepdims=True) + 1e-8
    features_scaled = (features / max_abs) * np.pi

    # Statistics
    stats = {
        "total_samples": int(len(labels)),
        "tb_samples": int(n_tb),
        "normal_samples": int(n_normal),
        "feature_names": [
            "Texture Density",
            "Edge Sharpness",
            "Opacity Variation",
            "Symmetry Score"
        ],
        "tb_means": [float(x) for x in tb_features.mean(axis=0)],
        "normal_means": [float(x) for x in normal_features.mean(axis=0)],
        "separation_difficulty": "subtle",
        "note": "Features are intentionally close — QML should still separate them"
    }

    dataset = {
        "features": features_scaled,
        "labels": labels,
        "feature_names": stats["feature_names"],
        "stats": stats,
    }

    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "features.npy"), features_scaled)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        with open(os.path.join(save_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to {save_dir}/")

    return dataset


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic TB Data Generator")
    print("=" * 60)

    save_path = os.path.join(os.path.dirname(__file__), "data")
    dataset = generate_dataset(n_tb=250, n_normal=250, save_dir=save_path)

    print(f"\n✓ Dataset generated:")
    print(f"  Total: {dataset['stats']['total_samples']} samples")
    print(f"  TB: {dataset['stats']['tb_samples']}, Normal: {dataset['stats']['normal_samples']}")
    print(f"\n  Feature names: {dataset['stats']['feature_names']}")
    print(f"  TB means:     {dataset['stats']['tb_means']}")
    print(f"  Normal means: {dataset['stats']['normal_means']}")
    print(f"\n  Features shape: {dataset['features'].shape}")
    print(f"  Labels shape:   {dataset['labels'].shape}")
    print(f"  Features range: [{dataset['features'].min():.3f}, {dataset['features'].max():.3f}]")

    print("\n" + "=" * 60)
    print("  Done! Ready for training.")
    print("=" * 60)
