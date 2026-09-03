"""
==========================================================================
  HOSPITAL-GRADE TB Chest X-Ray Model (11,000+ Images)
  Architecture: DenseNet-121
  Dataset: TBX11K (11,200 images)
==========================================================================
  Run this on Kaggle with GPU enabled (T4 x2 recommended).
  
  Instructions:
  1. Go to Kaggle Notebook -> Add Input
  2. Search for: "TBX11K Simplified" 
     (Or any TBX11K dataset that has organized folders)
  3. Run this script. It will take 1-2 hours to train on 11k images.
  4. Download 'tb_xray_densenet121_11k.pth'
==========================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json
from PIL import Image

print("=" * 70)
print("  Initializing 11,000+ Image TBX11K Training Pipeline")
print("=" * 70)

INPUT_DIR = "/kaggle/input"
dataset_path = None

import pandas as pd
import fnmatch

print("=" * 70)
print("  Initializing 11,000+ Image TBX11K Training Pipeline (CSV based)")
print("=" * 70)

INPUT_DIR = "/kaggle/input"
csv_path = None
images_dir = None
test_dir = None

print(f"Scanning directory: {INPUT_DIR}")
if os.path.exists(INPUT_DIR):
    for root, dirs, files in os.walk(INPUT_DIR):
        # Print what we are scanning to debug
        print(f"  -> In {root}: {len(dirs)} dirs, {len(files)} files")
        
        for file in files:
            if file.lower().endswith('.csv'):
                print(f"     - Found CSV: {file}")
                # We assume the first CSV we find is our metadata (usually data.csv)
                if not csv_path:
                    csv_path = os.path.join(root, file)
        
        lower_dirs = [d.lower() for d in dirs]
        if 'images' in lower_dirs and not images_dir:
            images_dir = os.path.join(root, 'images')
            print(f"     - Found images folder")
        if 'test' in lower_dirs and not test_dir:
            test_dir = os.path.join(root, 'test')

if not csv_path or not images_dir:
    print("\n❌ Could not find CSV or images folder. Current contents of /kaggle/input:")
    if os.path.exists(INPUT_DIR):
        for root, dirs, files in os.walk(INPUT_DIR):
            print(root)
            for f in files[:5]: print("  -", f)
    raise FileNotFoundError("Dataset structure not recognized. Ensure TBX11K Simplified is added.")

print(f"\n✅ Found CSV metadata at: {csv_path}")
print(f"✅ Found training images at: {images_dir}")
if test_dir:
    print(f"✅ Found test images at: {test_dir}")

# Read the CSV
df = pd.read_csv(csv_path)
print(f"  CSV Columns found: {df.columns.tolist()}")

# Identify filename and label columns dynamically
file_col = 'fname' if 'fname' in df.columns else None
label_col = 'target' if 'target' in df.columns else ('tb_type' if 'tb_type' in df.columns else None)

if not file_col or not label_col:
    print("  ⚠️ Could not find 'fname' or 'target' columns, falling back to heuristics...")
    for col in df.columns:
        c_lower = col.lower()
        if not file_col and ('file' in c_lower or 'image' in c_lower or 'path' in c_lower):
            file_col = col
        if not label_col and ('class' in c_lower or 'label' in c_lower or 'target' in c_lower or 'tb' in c_lower):
            label_col = col

print(f"  Using '{file_col}' for filenames and '{label_col}' for labels.")

# ======================================================================
# 1. DATA AUGMENTATION & DATALOADER
# ======================================================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Parse CSV to build a label dictionary
# If an image has multiple bounding boxes, it might appear multiple times in CSV
csv_labels = {}
for index, row in df.iterrows():
    fname = str(row[file_col]).strip()
    raw_label = str(row[label_col]).strip().lower()
    
    # If the label indicates TB (1, 'active tb', etc.)
    if raw_label == '1' or 'tb' in raw_label:
        csv_labels[fname] = 1
    elif fname not in csv_labels:
        # Only set to 0 if it hasn't already been marked as 1 by another row
        csv_labels[fname] = 0

# Now build the dataset by iterating over ALL image files on disk
dataset_items = []
n_tb = 0
n_normal = 0

for img_dir in [images_dir, test_dir]:
    if not img_dir: continue
    
    for f in os.listdir(img_dir):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        base_no_ext = os.path.splitext(f)[0]
        
        # Determine label from CSV
        if f in csv_labels:
            label = csv_labels[f]
        elif base_no_ext in csv_labels:
            label = csv_labels[base_no_ext]
        else:
            # If it's completely missing from the CSV, it means there are no bounding boxes
            # In TBX11K, images without bounding boxes are Healthy / Normal (0)
            label = 0
            
        img_path = os.path.join(img_dir, f)
        dataset_items.append((img_path, label))
        
        if label == 1:
            n_tb += 1
        else:
            n_normal += 1

print(f"\n  TOTAL UNIQUE IMAGES FOUND ON DISK: {len(dataset_items)}")
print(f"  Non-TB (Healthy/Sick): {n_normal} | Active TB: {n_tb}")

if len(dataset_items) == 0:
    raise ValueError("No images matched from CSV.")

class TBXrayDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Train/Val Split (80/20)
train_size = int(0.8 * len(dataset_items))
val_size = len(dataset_items) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    TBXrayDataset(dataset_items, transform=train_transform), 
    [train_size, val_size]
)

# Overwrite validation transform
val_ds.dataset.transform = val_transform

# Increase batch size for 11k images
BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ======================================================================
# 2. BUILD MODEL
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Using Device: {device}")

model = models.densenet121(weights='IMAGENET1K_V1')
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
)
model = model.to(device)

# For 11k images, we fine-tune MORE layers than before (DenseBlock 3 and 4)
for name, param in model.named_parameters():
    if 'denseblock3' in name or 'denseblock4' in name or 'classifier' in name or 'norm5' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ======================================================================
# 3. TRAIN
# ======================================================================

print("\n" + "=" * 70)
print("  Training on 11k Images (This will take ~1-2 hours)")
print("=" * 70)

# Class weight for imbalanced TBX11K dataset (Usually Normal >> TB)
pos_weight = torch.tensor([n_normal / max(1, n_tb)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

N_EPOCHS = 15
best_auc = 0.0

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    for i, (batch_imgs, batch_labels) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(batch_imgs).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if (i+1) % 50 == 0:
            print(f"    Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch_imgs, batch_labels in val_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.float().to(device)
            outputs = model(batch_imgs).squeeze()
            val_loss += criterion(outputs, batch_labels).item()
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
        
    scheduler.step(auc)
    
    print(f"  Epoch {epoch+1:2d}/{N_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val AUC: {auc*100:.1f}%")
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), '/kaggle/working/tb_xray_densenet121_11k.pth')
        print(f"    * New best model saved! (AUC: {auc*100:.1f}%)")

# ======================================================================
# 4. EXPORT
# ======================================================================

print("\n" + "=" * 70)
print("  Exporting Final Weights")
print("=" * 70)

# Extract features only
model.load_state_dict(torch.load('/kaggle/working/tb_xray_densenet121_11k.pth'))
state = model.state_dict()
feat_state = {k: v for k, v in state.items() if 'classifier' not in k}
torch.save(feat_state, "/kaggle/working/tb_xray_features_11k.pth")

print("\n  DONE! Download 'tb_xray_densenet121_11k.pth' from Output.")
