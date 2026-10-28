
from pathlib import Path
import random
import shutil

SOURCE = Path("raw/color")
DEST = Path("data")
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset folder not found: {SOURCE.resolve()}")

classes = sorted(p for p in SOURCE.iterdir() if p.is_dir())

if not classes:
    raise RuntimeError(
        f"No class folders found inside {SOURCE.resolve()}. "
        "Check the dataset folder structure."
    )

random.seed(SEED)

# Remove the previous split to avoid mixing old and new images.
if DEST.exists():
    shutil.rmtree(DEST)

total = 0

for class_dir in classes:
    images = [
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    random.shuffle(images)

    if len(images) < 3:
        print(f"Skipping {class_dir.name}: only {len(images)} images")
        continue

    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, files in splits.items():
        output_dir = DEST / split_name / class_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, source_file in enumerate(files):
            # Prefix filenames to prevent collisions.
            output_file = output_dir / f"{index}_{source_file.name}"
            shutil.copy2(source_file, output_file)

    total += n
    print(f"{class_dir.name}: {n} images")

print(f"\nTotal images processed: {total}")
print(f"Dataset splits saved to: {DEST.resolve()}")
print("Training: 70% | Validation: 15% | Testing: approximately 15%")
