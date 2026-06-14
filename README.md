# Deep Learning Project — CIFAR-10 Image Classifier (Transfer Learning)

## 📌 Overview
3-class image classifier (`airplane`, `cat`, `dog`) built on a CIFAR-10 subset using
**ResNet18** transfer learning in **PyTorch**, with data augmentation.

## 📁 Repository Structure
```
dl_project/
├── DL_Classification_Notebook.ipynb   # Main notebook (code + plots + metrics)
├── train.py                           # Standalone training script
├── predict.py                         # Inference CLI for new images
├── cifar_classifier.pth               # Saved trained model (~45 MB)
├── metrics.json                       # Final test metrics
├── training_curves.png                # Loss & accuracy curves
├── confusion_matrix.png               # Confusion matrix
├── per_class_metrics.png              # Precision/Recall/F1 per class
├── sample_augmented.png               # Example augmented training images
├── README.md                          # This file
└── data/                               # Dataset (see setup below)
```

## ⚙️ Environment Setup

### Requirements
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow numpy nbformat jupyter
```

### Dataset Setup
This project uses CIFAR-10 images organized in `ImageFolder` format
(`train/<class>/*.jpg`, `test/<class>/*.jpg`).

Download and extract:
```bash
cd dl_project
mkdir -p data && cd data
curl -L -o cifar.tar.gz "https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.tar.gz"
tar -xzf cifar.tar.gz
cd ..
```
This creates `data/CIFAR-10-images-master/train/<class>/...` and `data/CIFAR-10-images-master/test/<class>/...`
with all 10 CIFAR-10 classes (the script automatically selects `airplane`, `cat`, `dog`).

> **Note on pretrained weights:** `train.py` attempts to download ImageNet-pretrained
> ResNet18 weights automatically (`torchvision` downloads from `download.pytorch.org`
> on first run). If your network blocks this, the script automatically falls back to
> training the full network from scratch — no manual changes needed.

## ▶️ Run Training
```bash
python train.py
```
This will:
1. Load the 3-class CIFAR-10 subset (300 train / 100 test images per class)
2. Apply data augmentation (flip, rotation, color jitter, random crop)
3. Load pretrained ResNet18, replace + train the classifier head (or full network if no internet)
4. Train for 12 epochs, printing loss/accuracy per epoch
5. Generate all plots (`training_curves.png`, `confusion_matrix.png`, etc.)
6. Save the trained model to `cifar_classifier.pth`

## ▶️ Run the Notebook
```bash
jupyter notebook DL_Classification_Notebook.ipynb
```
Run all cells (`Run → Run All Cells`). Adjust `DATA_ROOT` at the top if your data
folder is in a different location.

## ▶️ Run Inference on New Images
```bash
python predict.py path/to/your_image.jpg
```
Multiple images:
```bash
python predict.py img1.jpg img2.png img3.jpg
```
With no arguments, it predicts on one sample image per class from the test set.

Output example:
```
Image: img1.jpg
  → Predicted: dog  (confidence: 84.45%)
  → All probabilities: {'airplane': 0.0005, 'cat': 0.1549, 'dog': 0.8445}
```

## 📊 Results Summary
See `metrics.json` for exact numbers. Reported metrics (macro-averaged):
- Accuracy
- Precision
- Recall
- F1 Score

The confusion matrix (`confusion_matrix.png`) shows that `cat` and `dog` are the
most commonly confused classes — expected, since both are quadrupeds and CIFAR-10
images are only 32x32 pixels.

## 🔬 Methodology
1. **Dataset:** CIFAR-10 subset — 3 classes (airplane, cat, dog), 300 train / 100 test images per class
2. **Augmentation:** Random horizontal flip, rotation (±10°), color jitter, random crop with padding
3. **Model:** ResNet18 (ImageNet-pretrained), backbone frozen, custom 2-layer head (128 units, dropout 0.3)
4. **Training:** Adam optimizer, CrossEntropy loss, StepLR scheduler (halve LR every 4 epochs), 12 epochs
5. **Evaluation:** Accuracy, macro Precision/Recall/F1, confusion matrix, per-class breakdown
