import os
import shutil
import random

# -----------------------------
# Import config
# -----------------------------
from config import RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, SPLIT_RATIO, CLASSES

# -----------------------------
# Prepare clean folders
# -----------------------------
def prepare_folders():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

# -----------------------------
# Split one category
# -----------------------------
def split_data(raw_path, class_name):
    # Ensure class subfolders exist in train/val/test
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        class_folder = os.path.join(folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Gather all images in raw folder
    files = [f for f in os.listdir(raw_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"⚠️ No images found for {class_name} in {raw_path}")
        return

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

    # Loop through each category in RAW_DIR
    for category in CLASSES:
        raw_category_path = os.path.join(RAW_DIR, category)
        if os.path.isdir(raw_category_path):
            split_data(raw_category_path, category)
        else:
            print(f"⚠️ Raw folder for {category} does not exist: {raw_category_path}")

    print("🎉 Dataset split complete! Check data/train, data/val, data/test.")
