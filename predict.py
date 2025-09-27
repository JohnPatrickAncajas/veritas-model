import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os

# ------------------------ 
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ 
# Config / Paths
# ------------------------
PROJECT_PATH = r"C:\Users\Patrick\Documents\GitHub\veritas-model"
PREDICT_ROOT = os.path.join(PROJECT_PATH, "predict")
PREDICT_FOLDER_NAME = "predict_friends"   # 👈 subfolder to predict
predict_folder = os.path.join(PREDICT_ROOT, PREDICT_FOLDER_NAME)

MODEL_PATH = os.path.join(PROJECT_PATH, "models", "efficientnet_ai_real_1k.pth")

# Classes
classes = ["ai", "real"]

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
# Predict images
# ------------------------
if not os.path.exists(predict_folder):
    raise FileNotFoundError(f"❌ The folder {predict_folder} does not exist!")

print(f"🔍 Predicting images from: {predict_folder}")
for filename in os.listdir(predict_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(predict_folder, filename)
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, 1).item()

        print(f"{filename}: Prediction -> {classes[pred]}")
print("✅ Prediction complete.")
