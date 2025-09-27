import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# ------------------------
# CONFIG
# ------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"
KAGGLE_CACHE = os.path.join(PROJECT_PATH, "kaggle_cache")

# Dataset links (easy to edit later)
AI_DATASET = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"
REAL_DATASET = "ciplab/real-and-fake-face-detection"

# ------------------------
# Setup Kaggle API
# ------------------------
os.environ['KAGGLE_CONFIG_DIR'] = PROJECT_PATH

api = KaggleApi()
api.authenticate()

# ------------------------
# Reset cache
# ------------------------
if os.path.exists(KAGGLE_CACHE):
    print("⚠️ Old cache found. Deleting...")
    shutil.rmtree(KAGGLE_CACHE)

os.makedirs(KAGGLE_CACHE, exist_ok=True)
print("📂 Fresh cache folder created:", KAGGLE_CACHE)

# ------------------------
# Download AI-generated faces
# ------------------------
print("⬇️ Downloading AI-generated faces dataset...")
api.dataset_download_files(
    AI_DATASET,
    path=KAGGLE_CACHE,
    unzip=True
)
print("✅ AI dataset cached.")

# ------------------------
# Download real human faces (new source)
# ------------------------
print("⬇️ Downloading Real human faces dataset (new source)...")
api.dataset_download_files(
    REAL_DATASET,
    path=KAGGLE_CACHE,
    unzip=True
)
print("✅ Real dataset cached.")

print("🎉 All datasets are now stored fresh in kaggle_cache.")
