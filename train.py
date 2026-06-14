"""
Deep Learning Project — CIFAR-10 Subset Image Classifier
Transfer Learning with ResNet18 (PyTorch)

Classes used (3-class subset for fast CPU training):
  - cat
  - dog
  - airplane
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUT_DIR = "/home/claude/dl_project"
DATA_ROOT = "/home/claude/dl_project/data/CIFAR-10-images-master"

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
SELECTED_CLASSES = ['airplane', 'cat', 'dog']   # CIFAR-10 subset (3 of 10 classes)
SAMPLES_PER_CLASS_TRAIN = 300   # 300 x 3 = 900 train images
SAMPLES_PER_CLASS_TEST  = 100   # 100 x 3 = 300 test images
BATCH_SIZE = 32
NUM_EPOCHS = 12
LR = 1e-3
IMG_SIZE = 64   # upscale 32x32 -> 64x64 for ResNet18

# ─────────────────────────────────────────────
# 2. DATA AUGMENTATION & TRANSFORMS
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────
# 3. LOAD DATA via ImageFolder + BUILD BALANCED SUBSET
# ─────────────────────────────────────────────
print("Loading CIFAR-10 images (ImageFolder)...")

full_train = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "train"))
full_test  = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "test"))

# class_to_idx maps class name -> index assigned by ImageFolder (alphabetical)
selected_orig_idx = {full_train.class_to_idx[c]: i for i, c in enumerate(SELECTED_CLASSES)}


class TransformedSubset(torch.utils.data.Dataset):
    """Wraps an ImageFolder, restricts to selected samples, remaps labels, applies transform."""
    def __init__(self, base_dataset, indices, transform, label_map):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, orig_label = self.base[self.indices[i]]
        img = self.transform(img)
        return img, self.label_map[orig_label]


def build_balanced_indices(dataset, per_class, selected_orig_idx_map):
    targets = np.array(dataset.targets)
    indices = []
    for orig_idx in selected_orig_idx_map.keys():
        cls_indices = np.where(targets == orig_idx)[0][:per_class]
        indices.extend(cls_indices.tolist())
    np.random.shuffle(indices)
    return indices


train_indices = build_balanced_indices(full_train, SAMPLES_PER_CLASS_TRAIN, selected_orig_idx)
test_indices  = build_balanced_indices(full_test,  SAMPLES_PER_CLASS_TEST,  selected_orig_idx)

train_dataset = TransformedSubset(full_train, train_indices, train_transform, selected_orig_idx)
test_dataset  = TransformedSubset(full_test,  test_indices,  test_transform,  selected_orig_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train set: {len(train_dataset)} images ({SAMPLES_PER_CLASS_TRAIN}/class x {len(SELECTED_CLASSES)})")
print(f"Test set : {len(test_dataset)} images ({SAMPLES_PER_CLASS_TEST}/class x {len(SELECTED_CLASSES)})")
print(f"Classes  : {SELECTED_CLASSES}")

# ─────────────────────────────────────────────
# 4. TRANSFER LEARNING MODEL — ResNet18
# ─────────────────────────────────────────────
print("\nLoading pretrained ResNet18 (ImageNet weights)...")
PRETRAINED_LOADED = True
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("Loaded ImageNet-pretrained weights.")
except Exception as e:
    print(f"Could not download pretrained weights ({e}).")
    print("Falling back to random initialization (architecture unchanged);")
    print("backbone will be trained from scratch instead of frozen.")
    model = models.resnet18(weights=None)
    PRETRAINED_LOADED = False

# Replace final fully-connected layer for our 3-class problem
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, len(SELECTED_CLASSES))
)

# Freeze backbone ONLY if pretrained weights loaded (true transfer learning).
# The new `fc` head is always trainable. If training from scratch, the
# entire network (including backbone) remains trainable.
if PRETRAINED_LOADED:
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False

model = model.to(DEVICE)

# Only train the new classifier head
# Optimizer: train only the new head if backbone is frozen (true transfer learning),
# otherwise train the full network (from-scratch fallback)
params_to_train = model.fc.parameters() if PRETRAINED_LOADED else model.parameters()
optimizer = optim.Adam(params_to_train, lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # ── Train ──
    model.train()
    running_loss, correct, total_samples = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = running_loss / total_samples
    train_acc  = correct / total_samples

    # ── Validate ──
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc  = val_correct / val_total

    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  |  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.1f}s")

# ─────────────────────────────────────────────
# 6. FINAL EVALUATION
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec  = recall_score(all_labels, all_preds, average='macro')
f1   = f1_score(all_labels, all_preds, average='macro')

print("\n" + "=" * 60)
print("FINAL TEST METRICS (macro-averaged)")
print("=" * 60)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=SELECTED_CLASSES)
print(report)

# Save metrics to JSON
metrics_summary = {
    "accuracy": round(float(acc), 4),
    "precision_macro": round(float(prec), 4),
    "recall_macro": round(float(rec), 4),
    "f1_macro": round(float(f1), 4),
    "classes": SELECTED_CLASSES,
    "train_size": len(train_dataset),
    "test_size": len(test_dataset),
    "epochs": NUM_EPOCHS,
    "training_time_seconds": round(elapsed, 1),
    "pretrained_imagenet_weights": PRETRAINED_LOADED,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

# Plot A: Training Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs_range = range(1, NUM_EPOCHS + 1)

axes[0].plot(epochs_range, history['train_loss'], 'o-', color='#4A90D9', label='Train Loss', lw=2)
axes[0].plot(epochs_range, history['val_loss'], 's-', color='#E67E22', label='Val Loss', lw=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)

axes[1].plot(epochs_range, history['train_acc'], 'o-', color='#4A90D9', label='Train Acc', lw=2)
axes[1].plot(epochs_range, history['val_acc'], 's-', color='#E67E22', label='Val Acc', lw=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_ylim(0, 1.05)

fig.suptitle('Training Curves — ResNet18 Transfer Learning', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# Plot B: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix — Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# Plot C: Per-class metrics bar chart
p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(SELECTED_CLASSES))
width = 0.25
ax.bar(x - width, p, width, label='Precision', color='#4A90D9')
ax.bar(x, r, width, label='Recall', color='#E67E22')
ax.bar(x + width, f, width, label='F1', color='#2ECC71')
ax.set_xticks(x)
ax.set_xticklabels(SELECTED_CLASSES, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: per_class_metrics.png")

# Plot D: Sample augmented images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
sample_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
images, labels = next(iter(sample_loader))
for i, ax in enumerate(axes.flat):
    img = images[i].numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(SELECTED_CLASSES[labels[i]], fontsize=11)
    ax.axis('off')
fig.suptitle('Sample Augmented Training Images', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sample_augmented.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sample_augmented.png")

# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': SELECTED_CLASSES,
    'img_size': IMG_SIZE,
    'mean': IMAGENET_MEAN,
    'std': IMAGENET_STD,
    'architecture': 'resnet18',
    'pretrained_imagenet_weights': PRETRAINED_LOADED,
}, f"{OUT_DIR}/cifar_classifier.pth")

print(f"\nSaved: cifar_classifier.pth ({os.path.getsize(f'{OUT_DIR}/cifar_classifier.pth')/1e6:.1f} MB)")
print("\n✅ Training pipeline completed successfully.")
