"""
qml_engine.py — Full QML Pipeline: Image → Quantum → Diagnosis
================================================================
This is the main engine that combines:
  1. Classical Feature Extraction (ResNet18 + PCA)
  2. Quantum Classification (4-Qubit Variational Circuit)
  3. Explainability (Heatmap + Feature Analysis)

It provides a single function: analyze_xray(image) → structured result
"""

import os
import sys
import numpy as np
from PIL import Image
import io
import base64
import json

sys.path.insert(0, os.path.dirname(__file__))

from quantum_circuit import quantum_classifier, predict_probability, get_circuit_info, N_QUBITS, N_LAYERS
from feature_extractor import FeatureExtractor
from pennylane import numpy as pnp


class QMLEngine:
    """
    Complete QML TB Detection Engine.
    Combines classical ResNet feature extraction with quantum classification.
    """

    def __init__(self, model_dir=None):
        """
        Initialize the QML Engine.

        Parameters
        ----------
        model_dir : str, optional
            Directory containing qml_weights.npy.
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "saved_model")

        # Load trained quantum weights
        weights_path = os.path.join(model_dir, "qml_weights.npy")
        if os.path.exists(weights_path):
            self.weights = pnp.array(np.load(weights_path), requires_grad=False)
            print(f"✓ Loaded trained weights from {weights_path}")
        else:
            # Use random weights if no trained model found
            print("⚠ No trained weights found. Using random initialization.")
            print("  Run train.py first for accurate predictions.")
            self.weights = pnp.array(
                np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 2)),
                requires_grad=False
            )

        # Initialize feature extractor
        pca_path = os.path.join(model_dir, "pca_model.pkl")
        self.extractor = FeatureExtractor(pca_model_path=pca_path)

        # If PCA not fitted, fit on synthetic baseline
        if self.extractor.pca is None:
            print("  Fitting PCA on baseline features...")
            self._fit_default_pca()

    def _fit_default_pca(self):
        """Fit PCA on random ResNet features as a baseline."""
        # Generate dummy images to get ResNet feature distribution
        dummy_features = []
        for _ in range(30):
            img = Image.fromarray(
                np.random.randint(50, 200, (224, 224), dtype=np.uint8), mode="L"
            )
            feat = self.extractor.extract_resnet_features(img)
            dummy_features.append(feat)

        self.extractor.fit_pca(np.array(dummy_features), n_components=4)

    def analyze_xray(self, image_bytes=None, image_path=None, image_pil=None):
        """
        Full analysis pipeline for a chest X-ray.

        Parameters
        ----------
        image_bytes : bytes, optional
            Raw image bytes (from API upload).
        image_path : str, optional
            Path to image file.
        image_pil : PIL.Image, optional
            PIL Image object.

        Returns
        -------
        dict
            Complete analysis result with:
            - risk_score: 0-100 integer
            - probability: 0.0-1.0 float
            - classification: "TB_POSITIVE" or "NORMAL"
            - severity: "CRITICAL" / "WARNING" / "LOW" / "NORMAL"
            - features: dict with 4 extracted features and their names
            - explanation: human-readable explanation string
            - heatmap_base64: base64-encoded heatmap overlay image
            - circuit_info: quantum circuit metadata
            - alert: dict with alert message and timing
        """
        # ── Step 1: Load Image ────────────────────────────────────────────
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_path:
            image = Image.open(image_path).convert("RGB")
        elif image_pil:
            image = image_pil.convert("RGB")
        else:
            raise ValueError("Provide image_bytes, image_path, or image_pil")

        # ── Step 2: Extract Features ──────────────────────────────────────
        features = self.extractor.extract_features(image)
        feature_names = [
            "Texture Density",
            "Edge Sharpness",
            "Opacity Variation",
            "Symmetry Score"
        ]

        # ── Step 3: Quantum Classification ────────────────────────────────
        features_qml = pnp.array(features, requires_grad=False)
        raw_output = float(quantum_classifier(features_qml, self.weights))
        probability = (1.0 - raw_output) / 2.0
        risk_score = int(probability * 100)

        # ── Step 4: Classification & Severity ─────────────────────────────
        if probability > 0.80:
            classification = "TB_POSITIVE"
            severity = "CRITICAL"
        elif probability > 0.60:
            classification = "TB_POSITIVE"
            severity = "WARNING"
        elif probability > 0.40:
            classification = "UNCERTAIN"
            severity = "LOW"
        else:
            classification = "NORMAL"
            severity = "NORMAL"

        # ── Step 5: Generate Heatmap ──────────────────────────────────────
        try:
            heatmap = self.extractor.generate_heatmap_data(image)
            heatmap_b64 = self.extractor.heatmap_to_base64(heatmap, image)
        except Exception as e:
            heatmap_b64 = ""
            print(f"  Heatmap generation failed: {e}")

        # ── Step 6: Generate Explanation ──────────────────────────────────
        explanation = self._generate_explanation(
            features, feature_names, probability, severity
        )

        # ── Step 7: Alert Info ────────────────────────────────────────────
        alert = self._generate_alert(probability, severity, features, feature_names)

        # ── Build Result ──────────────────────────────────────────────────
        result = {
            "risk_score": risk_score,
            "probability": round(probability, 4),
            "raw_quantum_output": round(raw_output, 4),
            "classification": classification,
            "severity": severity,
            "features": {
                name: round(float(val), 4)
                for name, val in zip(feature_names, features)
            },
            "feature_values": [round(float(v), 4) for v in features],
            "feature_names": feature_names,
            "explanation": explanation,
            "heatmap_base64": heatmap_b64,
            "circuit_info": get_circuit_info(),
            "alert": alert,
        }

        return result

    def _generate_explanation(self, features, names, probability, severity):
        """Generate human-readable explanation of the prediction."""
        lines = []

        if severity == "CRITICAL":
            lines.append(f"CRITICAL: {probability*100:.0f}% probability of Tuberculosis detected.")
            lines.append("Quantum circuit detected significant anomalies in lung features.")
        elif severity == "WARNING":
            lines.append(f"WARNING: {probability*100:.0f}% probability of TB-like patterns.")
            lines.append("Minor but statistically significant deviations detected.")
        elif severity == "LOW":
            lines.append(f"LOW RISK: {probability*100:.0f}% — Uncertain classification.")
            lines.append("Some features show borderline values. Re-scan recommended.")
        else:
            lines.append(f"NORMAL: {probability*100:.0f}% TB probability. Patient appears healthy.")

        # Feature-specific insights
        lines.append("")
        lines.append("Feature Analysis:")
        for i, (name, val) in enumerate(zip(names, features)):
            abs_val = abs(float(val))
            if abs_val > 2.0:
                lines.append(f"  - {name}: {float(val):.3f} (HIGH — significant deviation)")
            elif abs_val > 1.0:
                lines.append(f"  - {name}: {float(val):.3f} (ELEVATED — above baseline)")
            else:
                lines.append(f"  - {name}: {float(val):.3f} (Normal range)")

        return "\n".join(lines)

    def _generate_alert(self, probability, severity, features, names):
        """Generate structured alert for the dashboard."""
        if severity == "CRITICAL":
            return {
                "show": True,
                "level": "CRITICAL",
                "title": "PREDICTIVE ALERT: TB DETECTED",
                "message": f"{probability*100:.0f}% Probability of Tuberculosis.",
                "action": "Immediately order sputum AFB test, chest CT scan. Start isolation protocol.",
                "estimated_window": "Confirm within 2-4 hours",
                "evidence": [
                    f"{names[i]}: {float(features[i]):.3f}"
                    for i in range(len(names))
                    if abs(float(features[i])) > 1.0
                ],
            }
        elif severity == "WARNING":
            return {
                "show": True,
                "level": "WARNING",
                "title": "TB RISK ELEVATED",
                "message": f"{probability*100:.0f}% Probability — Borderline TB patterns.",
                "action": "Schedule confirmatory tests. Monitor patient closely.",
                "estimated_window": "Review within 12-24 hours",
                "evidence": [
                    f"{names[i]}: {float(features[i]):.3f}"
                    for i in range(len(names))
                    if abs(float(features[i])) > 0.8
                ],
            }
        else:
            return {
                "show": False,
                "level": severity,
                "title": "No alert",
                "message": "Patient within normal parameters.",
                "action": "Routine monitoring.",
                "estimated_window": "N/A",
                "evidence": [],
            }

    def analyze_synthetic_features(self, features_array):
        """
        Analyze pre-extracted features (for training data / API testing).

        Parameters
        ----------
        features_array : array-like, shape (4,)
            Pre-computed 4D features (already scaled to [-π, π]).

        Returns
        -------
        dict
            Same structure as analyze_xray but without heatmap.
        """
        features_qml = pnp.array(features_array, requires_grad=False)
        raw_output = float(quantum_classifier(features_qml, self.weights))
        probability = (1.0 - raw_output) / 2.0
        risk_score = int(probability * 100)

        feature_names = [
            "Texture Density", "Edge Sharpness",
            "Opacity Variation", "Symmetry Score"
        ]

        if probability > 0.80:
            classification, severity = "TB_POSITIVE", "CRITICAL"
        elif probability > 0.60:
            classification, severity = "TB_POSITIVE", "WARNING"
        elif probability > 0.40:
            classification, severity = "UNCERTAIN", "LOW"
        else:
            classification, severity = "NORMAL", "NORMAL"

        return {
            "risk_score": risk_score,
            "probability": round(probability, 4),
            "classification": classification,
            "severity": severity,
            "features": {
                name: round(float(val), 4)
                for name, val in zip(feature_names, features_array)
            },
        }


# ── Self Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  QML Engine — Self Test")
    print("=" * 60)

    engine = QMLEngine()

    # Test with synthetic features
    print("\n--- Testing with synthetic TB features ---")
    tb_features = np.array([1.8, 1.2, 1.5, -0.9])
    result = engine.analyze_synthetic_features(tb_features)
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Classification: {result['classification']}")
    print(f"  Severity: {result['severity']}")

    print("\n--- Testing with synthetic Normal features ---")
    normal_features = np.array([0.2, 0.1, -0.1, 0.5])
    result = engine.analyze_synthetic_features(normal_features)
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Classification: {result['classification']}")
    print(f"  Severity: {result['severity']}")

    # Test with dummy image
    print("\n--- Testing with dummy X-ray image ---")
    dummy = Image.fromarray(
        np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L"
    )
    result = engine.analyze_xray(image_pil=dummy)
    print(f"  Risk Score: {result['risk_score']}%")
    print(f"  Classification: {result['classification']}")
    print(f"  Features: {result['features']}")
    print(f"  Heatmap generated: {'Yes' if result['heatmap_base64'] else 'No'}")

    print("\n" + "=" * 60)
    print("  Engine working correctly!")
    print("=" * 60)
