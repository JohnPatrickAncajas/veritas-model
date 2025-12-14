# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import os
import numpy as np
from collections import Counter

# ---------------------
# Import config
# ---------------------
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    MODEL_NAME, MODEL_SAVE_DIR, DROPOUT_RATE
)

# ---------------------
# GPU Check
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("✅ GPU is available!")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available, using CPU.")
print("Using device:", device)

# ---------------------
# Transforms (Enhanced augmentation)
# ---------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize larger for random crop
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop for better generalization
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Random blur
    transforms.ToTensor(),
    normalize,
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# ---------------------
# Datasets & DataLoaders
# ---------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes (from config):", CLASSES)
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ---------------------
# Class Distribution & Weights
# ---------------------
# Calculate class distribution
class_counts = Counter([label for _, label in train_dataset.samples])
print("\nClass distribution in training set:")
for idx, cls in enumerate(CLASSES):
    count = class_counts.get(idx, 0)
    print(f"  {cls}: {count} images")

# Compute class weights for handling imbalance
if len(class_counts) > 0:
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (len(CLASSES) * class_counts.get(i, 1)) for i in range(len(CLASSES))]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
else:
    class_weights = None

# ---------------------
# Model
# ---------------------
model = EfficientNet.from_pretrained('efficientnet-b0')
# Add dropout for regularization
model._fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(model._fc.in_features, len(CLASSES))
)
model = model.to(device)

# ---------------------
# Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler - reduces LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ---------------------
# Training Loop with Early Stopping & Checkpointing
# ---------------------
best_val_acc = 0.0
best_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
patience = 5
patience_counter = 0
min_delta = 0.01  # Minimum improvement to reset patience

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / val_total * 100

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {epoch_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Model checkpointing - save best model
    if val_acc > best_val_acc + min_delta:
        best_val_acc = val_acc
        patience_counter = 0
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"⏳ No improvement for {patience_counter} epoch(s)")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        break

# ---------------------
# Load Best Model for Testing
# ---------------------
print("\n" + "="*60)
print("Loading best model for final evaluation...")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Loaded best model from {best_model_path}")
else:
    print("⚠️ Best model not found, using last epoch model")

# ---------------------
# Test Accuracy
# ---------------------
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
test_acc = test_correct / test_total * 100
print(f"\n📊 Final Test Accuracy: {test_acc:.2f}%")

# ---------------------
# Save Final Model
# ---------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}.pth")
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved to {final_model_path}")
print(f"✅ Best model saved to {best_model_path}")
print(f"\nBest validation accuracy achieved: {best_val_acc:.2f}%")
print("="*60)
