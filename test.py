import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

# ------------------------
# Import config
# ------------------------
from config import TEST_DIR, MODEL_SAVE_DIR, MODEL_NAME, CLASSES

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# Load model
# ------------------------
# Try to load the best model first, fallback to regular model
best_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
regular_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}.pth")

if os.path.exists(best_model_path):
    model_path = best_model_path
    print(f"✅ Loading best model: {best_model_path}")
elif os.path.exists(regular_model_path):
    model_path = regular_model_path
    print(f"⚠️ Best model not found, loading regular model: {regular_model_path}")
else:
    raise FileNotFoundError(f"❌ No model found in {MODEL_SAVE_DIR}")

# Use from_name to avoid downloading pretrained weights
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # ensures RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------------
# Prediction & Accuracy
# ------------------------
total_correct = 0
total_images = 0

for cls in CLASSES:
    folder_path = os.path.join(TEST_DIR, cls)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}, skipping...")
        continue

    correct = 0
    count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as image:
                    image = transform(image).unsqueeze(0).to(device)
            except Exception as e:
                print(f"⚠️ Failed to open {img_path}: {e}")
                continue

            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(output, 1).item()

            if pred == CLASSES.index(cls):
                correct += 1
            count += 1

    total_correct += correct
    total_images += count

    if count > 0:
        print(f"{cls.upper()} folder: Accuracy = {correct}/{count} ({correct/count*100:.2f}%)")
    else:
        print(f"{cls.upper()} folder: No images found, skipped.")

# Overall accuracy
if total_images > 0:
    print(f"\n📊 Overall Accuracy: {total_correct}/{total_images} ({total_correct/total_images*100:.2f}%)")
else:
    print("❌ No test images found in any class folders.")
