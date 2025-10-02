import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

# ---------------------
# Config import
# ---------------------
from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    CLASSES,
    IMG_SIZE,
    DEVICE,
    MODEL_DIR,
    MODEL_NAME,
)

if torch.cuda.is_available():
    print("✅ GPU is available!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available, using CPU.")

# ---------------------
# Transforms
# ---------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize,
])

val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    normalize,
])

# ---------------------
# Datasets & Loaders
# ---------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes (from dataset):", train_dataset.classes)
print("Expected classes (from config):", CLASSES)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ---------------------
# Model
# ---------------------
model = EfficientNet.from_pretrained("efficientnet-b0")
model._fc = nn.Linear(model._fc.in_features, len(CLASSES))
model = model.to(DEVICE)

# ---------------------
# Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------
# Training Loop
# ---------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = val_correct / val_total * 100

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {epoch_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# ---------------------
# Test accuracy
# ---------------------
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
test_acc = test_correct / test_total * 100
print(f"Test Accuracy: {test_acc:.2f}%")

# ---------------------
# Save model
# ---------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
