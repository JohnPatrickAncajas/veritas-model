import os
import shutil
import random

# -----------------------------
# Config / Editable variables
# -----------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"
KAGGLE_CACHE = os.path.join(PROJECT_PATH, "kaggle_cache")
RAW_AI_PATH = os.path.join(PROJECT_PATH, "data", "raw", "ai")
RAW_REAL_PATH = os.path.join(PROJECT_PATH, "data", "raw", "real")

# Number of images to select
NUM_AI_IMAGES = 1000
NUM_REAL_IMAGES = 1000

# -----------------------------
# Setup
# -----------------------------
# Clear old raw folders
for folder in [RAW_AI_PATH, RAW_REAL_PATH]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Function to select & copy images
# -----------------------------
def select_and_copy(base_cache_path, dest_folder, num_images, subfolder=None):
    base_path = base_cache_path if subfolder is None else os.path.join(base_cache_path, subfolder)

    files = []
    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))

    if len(files) == 0:
        print(f"⚠️ No images found in {base_path}. Check folder path!")
        return

    # Randomly select specified number of images
    selected_files = random.sample(files, min(num_images, len(files)))

    # Copy selected images to destination
    for f in selected_files:
        shutil.copy2(f, dest_folder)

    print(f"✅ {len(selected_files)} images copied to {dest_folder}")

# -----------------------------
# Select images
# -----------------------------
# For AI dataset
select_and_copy(KAGGLE_CACHE, RAW_AI_PATH, NUM_AI_IMAGES, subfolder="AI-face-detection-Dataset/AI")

# For Real dataset
select_and_copy(KAGGLE_CACHE, RAW_REAL_PATH, NUM_REAL_IMAGES, subfolder="real_and_fake_face/training_real")
