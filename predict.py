"""
Inference Script — CIFAR-10 Subset Image Classifier
Loads the saved model checkpoint and predicts the class of an input image.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image1.jpg path/to/image2.png ...

If no image path is given, runs on a few sample test images instead.
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar_classifier.pth")


def load_model(model_path=MODEL_PATH, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, len(checkpoint['classes']))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def get_transform(checkpoint):
    return transforms.Compose([
        transforms.Resize((checkpoint['img_size'], checkpoint['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint['mean'], checkpoint['std']),
    ])


def predict_image(image_path, model, checkpoint, transform, device="cpu"):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    classes = checkpoint['classes']
    result = {
        "image": image_path,
        "predicted_class": classes[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "all_probabilities": {c: round(probs[i].item(), 4) for i, c in enumerate(classes)},
    }
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model from {MODEL_PATH} ...")

    model, checkpoint = load_model(device=device)
    transform = get_transform(checkpoint)

    print(f"Classes: {checkpoint['classes']}")
    print(f"Pretrained ImageNet weights used during training: {checkpoint.get('pretrained_imagenet_weights', 'unknown')}")
    print("-" * 50)

    image_paths = sys.argv[1:]

    if not image_paths:
        # Fall back to a couple of sample images from the test set, if available
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data/CIFAR-10-images-master/test")
        if os.path.isdir(sample_dir):
            for cls in checkpoint['classes']:
                cls_dir = os.path.join(sample_dir, cls)
                if os.path.isdir(cls_dir):
                    files = sorted(os.listdir(cls_dir))[:1]
                    image_paths.extend(os.path.join(cls_dir, f) for f in files)
        if not image_paths:
            print("No image paths provided and no sample images found.")
            print("Usage: python predict.py path/to/image.jpg")
            return

    for path in image_paths:
        if not os.path.isfile(path):
            print(f"⚠️  File not found: {path}")
            continue
        result = predict_image(path, model, checkpoint, transform, device)
        print(f"Image: {result['image']}")
        print(f"  → Predicted: {result['predicted_class']}  (confidence: {result['confidence']:.2%})")
        print(f"  → All probabilities: {result['all_probabilities']}")
        print()


if __name__ == "__main__":
    main()
