import os
import shutil
import random

# -----------------------------
# Import central config
# -----------------------------
from config import RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, SPLIT_RATIO, CLASSES

# -----------------------------
# Helper: prepare clean folders
# -----------------------------
def prepare_folders():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

# -----------------------------
# Helper: split one category
# -----------------------------
def split_data(raw_path, class_name):
    # Ensure class subfolders exist
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        class_folder = os.path.join(folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Gather files
    files = [f for f in os.listdir(raw_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(SPLIT_RATIO[0] * n_total)
    n_val = int(SPLIT_RATIO[1] * n_total)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(TRAIN_DIR, class_name, f))
    for f in val_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(VAL_DIR, class_name, f))
    for f in test_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(TEST_DIR, class_name, f))

    print(f"✅ {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    prepare_folders()

    # Loop through only defined categories (so order matches config.CLASSES)
    for category in CLASSES:
        raw_path = os.path.join(RAW_DIR, category)
        if os.path.isdir(raw_path):
            split_data(raw_path, category)

    print("🎉 Data split complete! Ready for ImageFolder.")
