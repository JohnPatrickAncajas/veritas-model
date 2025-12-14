import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

# ------------------------
# Config Import
# ------------------------
from config import (
    DEVICE,
    MODEL_PATH,
    CLASSES,
    PREDICT_DIR,
    DROPOUT_RATE,
)

# ------------------------
# Load model
# ------------------------
model = EfficientNet.from_pretrained("efficientnet-b0")
# Match the training architecture with dropout
model._fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(model._fc.in_features, len(CLASSES))
)
# Prefer the best model if available
from config import MODEL_SAVE_DIR, MODEL_NAME
best_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
if os.path.exists(best_model_path):
    model_path_to_load = best_model_path
    print(f"✅ Loading best model: {best_model_path}")
else:
    model_path_to_load = MODEL_PATH
    print(f"⚠️ Best model not found, loading regular model: {MODEL_PATH}")

model.load_state_dict(torch.load(model_path_to_load, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------
# Predict images
# ------------------------
if not os.path.exists(PREDICT_DIR):
    raise FileNotFoundError(f"❌ The folder {PREDICT_DIR} does not exist!")

print(f"🔍 Predicting images from: {PREDICT_DIR}")
for filename in os.listdir(PREDICT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img_path = os.path.join(PREDICT_DIR, filename)
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, 1).item()

        print(f"{filename}: Prediction -> {CLASSES[pred]}")
print("✅ Prediction complete.")
