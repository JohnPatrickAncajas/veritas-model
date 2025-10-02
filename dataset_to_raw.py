import os
import shutil
import random

# -----------------------------
# Config import
# -----------------------------
from config import (
    PROJECT_PATH,
    KAGGLE_CACHE,
    RAW_DIR,
    CLASSES,
    MAX_RAW_IMAGES,
)

# Map dataset sources for each class (using config.CLASSES order)
CATEGORIES = {
    "2d": "2D",
    "3d": "3D",
    "ai": "AI-face-detection-Dataset/AI",
    "real": "real_and_fake_face/training_real",
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

    # Randomly select specified number of images
    selected_files = random.sample(files, min(num_images, len(files)))

    # Copy selected images to destination
    for f in selected_files:
        shutil.copy2(f, dest_folder)

    print(f"✅ {len(selected_files)} images copied to {dest_folder}")
    return len(selected_files)

# -----------------------------
# Balanced allocation
# -----------------------------
per_class = MAX_RAW_IMAGES // len(CLASSES)
total_selected = 0

for category in CLASSES:
    subfolder = CATEGORIES[category]
    dest_folder = os.path.join(RAW_DIR, category)
    copied = select_and_copy(KAGGLE_CACHE, subfolder, dest_folder, per_class)
    total_selected += copied

print(f"🎉 Total images copied: {total_selected} (limit {MAX_RAW_IMAGES})")
