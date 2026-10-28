"""Predict the class of one image from the command line."""

from pathlib import Path
import argparse
import json

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

MODEL_PATH = Path("artifacts/best_model.pth")
CLASSES_PATH = Path("artifacts/class_names.json")

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to a plant leaf image")
args = parser.parse_args()

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    raise FileNotFoundError("Train the model first using python train.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(args.image).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    confidence, index = probabilities.max(dim=0)

print("Predicted class:", class_names[index.item()])
print(f"Model confidence: {confidence.item():.2%}")
