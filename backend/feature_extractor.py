"""
feature_extractor.py — Classical Feature Extraction from Chest X-Rays
=====================================================================
Uses a pre-trained ResNet18 (from torchvision) to extract high-level features
from chest X-ray images, then reduces them to 4 features using PCA.

Pipeline:
  Raw X-Ray Image (any size)
    → Resize to 224×224, normalize
    → ResNet18 (pre-trained on ImageNet, last layer removed)
    → 512-dimensional feature vector
    → PCA reduction → 4 features
    → These 4 numbers become qubit rotation angles

Why ResNet18?
  - Free, pre-trained, no additional training needed
  - Excellent at extracting texture, edge, and density features
  - These features capture subtle patterns like minor scarring,
    density variations, and cavity edges in lung tissue

Why PCA to 4?
  - Our quantum circuit has 4 qubits
  - PCA preserves the most variance (information) in fewer dimensions
  - 4 principal components typically capture 85%+ of meaningful variation
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import io
import base64
import os
import pickle


class FeatureExtractor:
    """
    Extract 4 principal features from a chest X-ray image using
    ResNet18 + PCA.
    """

    def __init__(self, pca_model_path=None):
        """
        Initialize the feature extractor.

        Parameters
        ----------
        pca_model_path : str, optional
            Path to a pre-fitted PCA model. If None, PCA will need to be
            fitted on training data first.
        """
        # ── ResNet18 Setup ────────────────────────────────────────────────
        # Load pre-trained ResNet18 and remove the final classification layer.
        # This gives us a 512-dim feature vector per image.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the last fully connected layer (classifier)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()  # Set to evaluation mode (no dropout, batch norm fixed)

        # ── Image Preprocessing ───────────────────────────────────────────
        # Standard ImageNet preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            ),
        ])

        # ── PCA Setup ────────────────────────────────────────────────────
        self.pca = None
        if pca_model_path and os.path.exists(pca_model_path):
            with open(pca_model_path, "rb") as f:
                self.pca = pickle.load(f)

    def extract_resnet_features(self, image):
        """
        Extract 512-dim feature vector from a PIL Image using ResNet18.

        Parameters
        ----------
        image : PIL.Image
            Input chest X-ray image.

        Returns
        -------
        np.ndarray, shape (512,)
            Feature vector from ResNet18's penultimate layer.
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Extract features (no gradient computation needed)
        with torch.no_grad():
            features = self.resnet(img_tensor)

        # Flatten from (1, 512, 1, 1) to (512,)
        return features.squeeze().numpy()

    def fit_pca(self, feature_matrix, n_components=4):
        """
        Fit PCA on a matrix of ResNet features.

        Parameters
        ----------
        feature_matrix : np.ndarray, shape (n_samples, 512)
            Matrix of ResNet features from multiple images.
        n_components : int
            Number of principal components (default 4 for 4 qubits).
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(feature_matrix)
        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"  PCA fitted: {n_components} components explain {explained:.1f}% variance")

    def save_pca(self, path):
        """Save the fitted PCA model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)

    def extract_features(self, image):
        """
        Full pipeline: Image → ResNet18 → PCA → 4 features.

        Parameters
        ----------
        image : PIL.Image
            Input chest X-ray image.

        Returns
        -------
        np.ndarray, shape (4,)
            4 principal features, scaled to [-π, π] for qubit angle embedding.
        """
        # Step 1: Get 512-dim ResNet features
        resnet_feat = self.extract_resnet_features(image)

        # Step 2: Reduce to 4 dimensions via PCA
        if self.pca is None:
            raise RuntimeError("PCA model not fitted. Call fit_pca() first or load a saved PCA model.")
        reduced = self.pca.transform(resnet_feat.reshape(1, -1))[0]

        # Step 3: Scale features to [-π, π] for angle embedding
        # Normalize to [-1, 1] then scale to [-π, π]
        max_abs = np.max(np.abs(reduced)) + 1e-8  # prevent division by zero
        scaled = (reduced / max_abs) * np.pi

        return scaled.astype(np.float64)

    def extract_features_from_bytes(self, image_bytes):
        """Extract features from raw image bytes (for API upload)."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.extract_features(image)

    def generate_heatmap_data(self, image):
        """
        Generate a simple activation-based heatmap showing which regions
        of the X-ray contributed most to the features.

        This uses the intermediate activations of ResNet's last conv layer
        to create a spatial attention map (simplified GradCAM).

        Parameters
        ----------
        image : PIL.Image
            Input chest X-ray image.

        Returns
        -------
        np.ndarray, shape (7, 7)
            Normalized activation map (0-1 range).
        """
        img_tensor = self.transform(image).unsqueeze(0)

        # Get the last convolutional layer's output
        # ResNet18: children()[:-2] gives us up to the last conv block
        conv_model = nn.Sequential(*list(
            models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children()
        )[:-2])
        conv_model.eval()

        with torch.no_grad():
            activation = conv_model(img_tensor)

        # Average across channels to get spatial attention map
        heatmap = activation.squeeze().mean(dim=0).numpy()

        # Normalize to 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def heatmap_to_base64(self, heatmap, original_image):
        """
        Overlay heatmap on original image and return as base64 string.

        Parameters
        ----------
        heatmap : np.ndarray
            Activation heatmap from generate_heatmap_data().
        original_image : PIL.Image
            Original X-ray image.

        Returns
        -------
        str
            Base64-encoded PNG image of the overlay.
        """
        from PIL import ImageDraw

        # Resize heatmap to match image
        img = original_image.copy().convert("RGBA")
        w, h = img.size

        # Create colored overlay
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Map heatmap values to colored rectangles
        hm_h, hm_w = heatmap.shape
        cell_w, cell_h = w / hm_w, h / hm_h

        for i in range(hm_h):
            for j in range(hm_w):
                val = heatmap[i, j]
                # Red channel intensity proportional to activation
                r = int(255 * val)
                g = int(50 * (1 - val))
                alpha = int(120 * val)
                x0, y0 = int(j * cell_w), int(i * cell_h)
                x1, y1 = int((j + 1) * cell_w), int((i + 1) * cell_h)
                draw.rectangle([x0, y0, x1, y1], fill=(r, g, 0, alpha))

        # Composite
        result = Image.alpha_composite(img, overlay).convert("RGB")

        # To base64
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Feature Extractor — Self Test")
    print("=" * 60)

    extractor = FeatureExtractor()
    print("✓ ResNet18 loaded successfully")

    # Create a dummy grayscale image (simulating an X-ray)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L")
    print("✓ Created dummy 256×256 grayscale image")

    # Extract ResNet features
    resnet_feat = extractor.extract_resnet_features(dummy_img)
    print(f"✓ ResNet features: shape={resnet_feat.shape}, range=[{resnet_feat.min():.3f}, {resnet_feat.max():.3f}]")

    # Fit PCA on a batch of dummy features
    dummy_batch = np.random.randn(50, 512)  # 50 samples
    extractor.fit_pca(dummy_batch, n_components=4)
    print("✓ PCA fitted on 50 dummy samples")

    # Full pipeline
    features = extractor.extract_features(dummy_img)
    print(f"✓ Final 4 features: {features}")
    print(f"  Range: [{features.min():.3f}, {features.max():.3f}] (target: [-π, π])")

    # Heatmap
    heatmap = extractor.generate_heatmap_data(dummy_img)
    print(f"✓ Heatmap generated: shape={heatmap.shape}, range=[{heatmap.min():.2f}, {heatmap.max():.2f}]")

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
