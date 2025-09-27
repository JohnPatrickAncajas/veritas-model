import os
import shutil
import random

# -----------------------------
# Config
# -----------------------------
RAW_AI_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model\data\raw\ai"
RAW_REAL_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model\data\raw\real"

TRAIN_DIR = r"C:\Users\Patrick\Documents\GitHub\veritas-model\data\train"
VAL_DIR = r"C:\Users\Patrick\Documents\GitHub\veritas-model\data\val"
TEST_DIR = r"C:\Users\Patrick\Documents\GitHub\veritas-model\data\test"

SPLIT_RATIO = (0.7, 0.2, 0.1)  # train, val, test

# -----------------------------
# Helper function to split
# -----------------------------
def split_data(raw_path, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    class_name = os.path.basename(raw_path)
    # Clear and recreate destination folders
    for folder in [train_dir, val_dir, test_dir]:
        class_folder = os.path.join(folder, class_name)
        if os.path.exists(class_folder):
            shutil.rmtree(class_folder)
        os.makedirs(class_folder, exist_ok=True)

    files = [f for f in os.listdir(raw_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)

    n_total = len(files)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(train_dir, class_name, f))
    for f in val_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(val_dir, class_name, f))
    for f in test_files:
        shutil.copy2(os.path.join(raw_path, f), os.path.join(test_dir, class_name, f))

    print(f"✅ {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images.")

# -----------------------------
# Run split for AI and Real
# -----------------------------
split_data(RAW_AI_PATH, TRAIN_DIR, VAL_DIR, TEST_DIR)
split_data(RAW_REAL_PATH, TRAIN_DIR, VAL_DIR, TEST_DIR)
