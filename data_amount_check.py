import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = r"D:\Programming\ProgrammingProjects\Veritas_DA_Project\veritas-model\data"
SETS = ["raw", "train", "val", "test"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# -----------------------------
# Function to count images
# -----------------------------
def count_images(folder_path):
    counts = {}
    if not os.path.exists(folder_path):
        return counts
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ])
            counts[class_name] = num_images
    return counts

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    for dataset in SETS:
        dataset_path = os.path.join(BASE_DIR, dataset)
        counts = count_images(dataset_path)
        total = sum(counts.values())
        print(f"\n📁 {dataset.upper()} folder: total images = {total}")
        for class_name, num in counts.items():
            print(f"   {class_name}: {num}")
