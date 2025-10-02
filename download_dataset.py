import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# ------------------------
# CONFIG
# ------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"

# Separate cache
KAGGLE_CACHE = os.path.join(PROJECT_PATH, "kaggle_cache")

# Kaggle datasets
AI_DATASET = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"
REAL_DATASET = "ciplab/real-and-fake-face-detection"

# Final dataset structure for training
FINAL_DATASET_DIR = os.path.join(PROJECT_PATH, "data")
os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

# ------------------------
# Setup Kaggle API
# ------------------------
os.environ['KAGGLE_CONFIG_DIR'] = PROJECT_PATH  # Folder where kaggle.json lives
api = KaggleApi()
api.authenticate()

# ------------------------
# Reset cache
# ------------------------
if os.path.exists(KAGGLE_CACHE):
    print(f"⚠️ Old Kaggle cache found at {KAGGLE_CACHE}. Deleting...")
    shutil.rmtree(KAGGLE_CACHE)
os.makedirs(KAGGLE_CACHE, exist_ok=True)
print(f"📂 Fresh Kaggle cache folder created: {KAGGLE_CACHE}")

# ------------------------
# Download Kaggle datasets
# ------------------------
print("⬇️ Downloading AI-generated faces dataset (Kaggle)...")
api.dataset_download_files(AI_DATASET, path=KAGGLE_CACHE, unzip=True)
print("✅ AI dataset cached.")

print("⬇️ Downloading Real human faces dataset (Kaggle)...")
api.dataset_download_files(REAL_DATASET, path=KAGGLE_CACHE, unzip=True)
print("✅ Real dataset cached.")

# ------------------------
# Summary / instructions
# ------------------------
print("🎉 All Kaggle datasets are now stored in the cache:")
print(f"   Kaggle datasets: {KAGGLE_CACHE}")
print("\n💡 Make sure each class (ai, real) is inside its own folder for training.")
