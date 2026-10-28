"""Train a ResNet18 transfer-learning model for plant disease classification."""

from pathlib import Path
import copy
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# ---------------- CONFIGURATION ----------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASETS ----------------
train_path = DATA_DIR / "train"
val_path = DATA_DIR / "val"
test_path = DATA_DIR / "test"

for path in [train_path, val_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run python split_dataset.py first."
        )

train_data = datasets.ImageFolder(train_path, transform=train_transform)
val_data = datasets.ImageFolder(val_path, transform=eval_transform)
test_data = datasets.ImageFolder(test_path, transform=eval_transform)

if not train_data.classes:
    raise RuntimeError("No training classes found.")

if (train_data.class_to_idx != val_data.class_to_idx
        or train_data.class_to_idx != test_data.class_to_idx):
    raise RuntimeError("Class folders differ between dataset splits.")

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", num_classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

# num_workers=0 is reliable on Windows.
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ---------------- MODEL ----------------
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace final layer for the plant disease classes.
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_val_accuracy = 0.0
best_state = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2%}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

        torch.save(best_state, OUTPUT_DIR / "best_model.pth")
        with open(OUTPUT_DIR / "class_names.json", "w") as f:
            json.dump(class_names, f, indent=2)

# Restore best validation model.
model.load_state_dict(best_state)
model.eval()

# ---------------- TEST EVALUATION ----------------
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(1)

        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

test_accuracy = test_correct / test_total
print(f"\nBest validation accuracy: {best_val_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")

with open(OUTPUT_DIR / "history.json", "w") as f:
    json.dump(history, f, indent=2)

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump({
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "num_classes": num_classes,
        "test_images": test_total
    }, f, indent=2)

# ---------------- PLOTS ----------------
epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, history["train_accuracy"], label="Train accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy.png")
plt.close()

plt.figure()
plt.plot(epochs, history["train_loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss.png")
plt.close()

print("\nTraining complete. Files saved in:", OUTPUT_DIR.resolve())
