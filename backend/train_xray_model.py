"""
==========================================================================
  REAL TB Chest X-Ray Model — DenseNet-121 Fine-Tuned on TB Datasets
==========================================================================
  Run this on Kaggle with GPU enabled.
  
  Dataset needed: "kmader/pulmonary-chest-xray-abnormalities"
  (Add this dataset to your Kaggle notebook)
  
  This trains a GENUINE medical AI model that:
  1. Learns to distinguish TB vs Normal chest X-rays
  2. Can extract zone-level pathology features
  3. Exports weights for our QML pipeline
==========================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json

import pandas as pd
import fnmatch

# ======================================================================
# 1. LOAD THE DATASETS (Tapendu Karmakar Dataset - CSV based)
# ======================================================================

print("=" * 70)
print("  Loading TB Chest X-Ray Dataset (MetaData.csv based)")
print("=" * 70)

INPUT_DIR = "/kaggle/input"
csv_path = None
images_dir = None

# Find MetaData.csv and the directory containing images
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.lower() == 'metadata.csv':
            csv_path = os.path.join(root, file)
        # Check if this directory contains a lot of images
        if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
            if images_dir is None:
                images_dir = root

if not csv_path:
    raise FileNotFoundError("Could not find 'MetaData.csv' in the dataset. Please ensure the Tapendu Karmakar dataset is added.")

if not images_dir:
    raise FileNotFoundError("Could not find a directory containing images.")

print(f"✅ Found CSV metadata at: {csv_path}")
print(f"✅ Found images at: {images_dir}")

# Read the CSV
df = pd.read_csv(csv_path)

# Ensure columns exist
if 'id' not in df.columns or 'ptb' not in df.columns:
    # Some versions might have different column names, try to guess
    cols = [c.lower() for c in df.columns]
    if 'id' not in cols or 'ptb' not in cols:
        print(f"CSV Columns found: {df.columns.tolist()}")
        raise ValueError("MetaData.csv does not have the expected 'id' and 'ptb' columns.")

# Get list of all image files available
all_image_files = os.listdir(images_dir)

images = []
n_tb = 0
n_normal = 0
missing_images = 0

for index, row in df.iterrows():
    img_id = str(row['id']).strip()
    label = int(row['ptb'])
    
    # Try to find the image file that matches this ID
    # Filenames might be like "1000.png", "CHNCXR_1000_0.png", etc.
    matched_file = None
    
    # First try exact match with extensions
    for ext in ['.png', '.jpg', '.jpeg']:
        if (img_id + ext) in all_image_files:
            matched_file = img_id + ext
            break
            
    # If not found, try substring match (e.g. ID is 1000, filename is CHNCXR_1000_0.png)
    if not matched_file:
        for f in all_image_files:
            if img_id in f:
                matched_file = f
                break
                
    if matched_file:
        img_path = os.path.join(images_dir, matched_file)
        images.append((img_path, label))
        if label == 1:
            n_tb += 1
        else:
            n_normal += 1
    else:
        missing_images += 1

print(f"\n  TOTAL MATCHED IMAGES: {len(images)}")
print(f"  Normal: {n_normal} | TB: {n_tb}")
if missing_images > 0:
    print(f"  ⚠️ Could not find image files for {missing_images} IDs in the CSV.")

if len(images) == 0:
    raise ValueError("No images were successfully matched with the CSV metadata. Cannot proceed.")

# ======================================================================
# 2. DATASET & DATALOADER
# ======================================================================

class TBXrayDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        path, label = self.image_list[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42, stratify=[l for _, l in images])
print(f"\n  Train: {len(train_imgs)} | Val: {len(val_imgs)}")

train_ds = TBXrayDataset(train_imgs, train_transform)
val_ds = TBXrayDataset(val_imgs, val_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ======================================================================
# 3. BUILD MODEL
# ======================================================================

print("\n" + "=" * 70)
print("  Building DenseNet-121 for TB Detection")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = models.densenet121(weights='IMAGENET1K_V1')
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),
)
model = model.to(device)

# Freeze early layers, fine-tune last dense block + classifier
for name, param in model.named_parameters():
    if 'denseblock4' not in name and 'classifier' not in name and 'norm5' not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ======================================================================
# 4. TRAIN
# ======================================================================

print("\n" + "=" * 70)
print("  Training...")
print("=" * 70)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

N_EPOCHS = 15
best_auc = 0.0

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(batch_imgs).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch_imgs, batch_labels in val_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.float().to(device)
            outputs = model(batch_imgs).squeeze()
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            all_labels.extend(batch_labels.cpu().numpy().tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
          f"Val: {val_loss/len(val_loader):.4f} | Acc: {acc*100:.1f}% | F1: {f1*100:.1f}% | AUC: {auc*100:.1f}%")
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), '/kaggle/working/best_tb_xray_model.pth')
        print(f"    * New best! AUC: {auc*100:.1f}%")
    
    scheduler.step()

# ======================================================================
# 5. FINAL EVALUATION
# ======================================================================

print("\n" + "=" * 70)
print("  Final Evaluation")
print("=" * 70)

model.load_state_dict(torch.load('/kaggle/working/best_tb_xray_model.pth'))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch_imgs, batch_labels in val_loader:
        batch_imgs = batch_imgs.to(device)
        outputs = model(batch_imgs).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
        all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
        all_labels.extend(batch_labels.numpy().tolist())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"  Accuracy : {acc*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  AUC-ROC  : {auc*100:.2f}%")
print(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

# ======================================================================
# 6. EXPORT
# ======================================================================

print("\n" + "=" * 70)
print("  Exporting for QML Pipeline")
print("=" * 70)

save_dir = "/kaggle/working"
torch.save(model.state_dict(), os.path.join(save_dir, "tb_xray_densenet121.pth"))

# Save feature extractor weights separately
state = model.state_dict()
feat_state = {k: v for k, v in state.items() if 'classifier' not in k}
torch.save(feat_state, os.path.join(save_dir, "tb_xray_features.pth"))

results = {
    "model": "DenseNet-121 fine-tuned on Shenzhen+Montgomery TB datasets",
    "total_images": len(images),
    "n_normal": n_normal,
    "n_tb": n_tb,
    "epochs": N_EPOCHS,
    "best_auc": round(best_auc * 100, 2),
    "final_accuracy": round(acc * 100, 2),
    "final_f1": round(f1 * 100, 2),
    "final_auc": round(auc * 100, 2),
}
with open(os.path.join(save_dir, "tb_xray_training_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Files saved:")
for fname in ["tb_xray_densenet121.pth", "tb_xray_features.pth", "tb_xray_training_results.json"]:
    fpath = os.path.join(save_dir, fname)
    if os.path.exists(fpath):
        print(f"    - {fname} ({os.path.getsize(fpath)/1e6:.1f} MB)")

print("\n  NEXT STEPS:")
print("  1. Download 'tb_xray_densenet121.pth' from Kaggle Output")
print("  2. Place in: QML_Report/backend/saved_model/")
print("  3. Backend will auto-load it for real X-ray zone analysis")
