import os
import shutil
import random

# -----------------------------
# Config import
# -----------------------------
from config import (
    PROJECT_PATH,
    CLASSES,
    MAX_RAW_IMAGES,
)

# Set RAW_DIR inside data/raw
RAW_DIR = os.path.join(PROJECT_PATH, "data", "raw")

# Dataset folder path
DATASET_DIR = os.path.join(PROJECT_PATH, "dataset")

# Map dataset sources for each class (now from dataset folder)
CATEGORIES = {
    "2d": "2d_dataset",
    "3d": "3D_dataset",
    "ai": "AI_dataset",
    "real": "Real_dataset",
}

# -----------------------------
# Setup: clear old raw folders
# -----------------------------
if os.path.exists(RAW_DIR):
    shutil.rmtree(RAW_DIR)
os.makedirs(RAW_DIR, exist_ok=True)

for category in CLASSES:
    os.makedirs(os.path.join(RAW_DIR, category), exist_ok=True)

# -----------------------------
# Function to select & copy images
# -----------------------------
def select_and_copy(dataset_folder, dest_folder, num_images):
    base_path = os.path.join(DATASET_DIR, dataset_folder)

    files = []
    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))

    if len(files) == 0:
        print(f"⚠️ No images found in {base_path}. Check folder path!")
        return 0

    # Shuffle and pick specified number of images
    selected_files = random.sample(files, min(num_images, len(files)))

    # Copy selected images to destination folder
    for f in selected_files:
        shutil.copy2(f, dest_folder)

    print(f"✅ {len(selected_files)} images copied to {dest_folder}")
    return len(selected_files)

# -----------------------------
# Balanced allocation: MAX_RAW_IMAGES per class
# -----------------------------
total_selected = 0

for category in CLASSES:
    dataset_folder = CATEGORIES[category]
    dest_folder = os.path.join(RAW_DIR, category)
    copied = select_and_copy(dataset_folder, dest_folder, MAX_RAW_IMAGES)
    total_selected += copied

print(f"🎉 Total images copied: {total_selected} (max {MAX_RAW_IMAGES * len(CLASSES)})")
