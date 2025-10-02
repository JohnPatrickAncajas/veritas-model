import os
import shutil
import random

# -----------------------------
# Config import
# -----------------------------
from config import (
    PROJECT_PATH,
    KAGGLE_CACHE,
    GOOGLE_CACHE,
    CLASSES,
    MAX_RAW_IMAGES,
)

# Set RAW_DIR inside data/raw
RAW_DIR = os.path.join(PROJECT_PATH, "data", "raw")

# Map dataset sources for each class
CATEGORIES = {
    "2d": "2D",                                  # From google_cache
    "3d": "3D",                                  # From google_cache (may have subfolders)
    "ai": "AI-face-detection-Dataset/AI",        # From kaggle_cache
    "real": "real_and_fake_face/training_real",  # From kaggle_cache
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
def select_and_copy(base_cache_path, subfolder, dest_folder, num_images):
    base_path = os.path.join(base_cache_path, subfolder)

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
    subfolder = CATEGORIES[category]

    # Use google_cache for 2d/3d, kaggle_cache for ai/real
    base_cache = GOOGLE_CACHE if category in ["2d", "3d"] else KAGGLE_CACHE

    dest_folder = os.path.join(RAW_DIR, category)
    copied = select_and_copy(base_cache, subfolder, dest_folder, MAX_RAW_IMAGES)
    total_selected += copied

print(f"🎉 Total images copied: {total_selected} (max {MAX_RAW_IMAGES * len(CLASSES)})")
