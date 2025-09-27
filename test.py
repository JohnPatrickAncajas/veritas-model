import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

# ------------------------
# CONFIG
# ------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"
TEST_DIR = os.path.join(PROJECT_PATH, "data", "test")
MODEL_PATH = os.path.join(PROJECT_PATH, "models", "efficientnet_ai_real_1k.pth")

# Classes (must match your folder names in test/)
classes = ["ai", "real"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# Load model
# ------------------------
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # ensures 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------------
# Prediction & Accuracy
# ------------------------
total_correct = 0
total_images = 0

for cls in classes:
    folder_path = os.path.join(TEST_DIR, cls)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}, skipping...")
        continue

    correct = 0
    count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(output, 1).item()

            if pred == classes.index(cls):
                correct += 1
            count += 1

    total_correct += correct
    total_images += count
    print(f"{cls.upper()} folder: Accuracy = {correct}/{count} ({correct/count*100:.2f}%)")

# Overall accuracy
if total_images > 0:
    print(f"\nOverall Accuracy: {total_correct}/{total_images} ({total_correct/total_images*100:.2f}%)")
