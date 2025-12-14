import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from efficientnet_pytorch import EfficientNet

# ================= CONFIGURATION =================
# Import configuration from config.py
from config import (
    TEST_DIR,
    MODEL_SAVE_DIR,
    MODEL_NAME,
    CLASSES,
    DEVICE,
    DROPOUT_RATE
)

BATCH_SIZE = 32
IMG_SIZE = 224

# Construct the best model path
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth") 

# Use device from config
print(f"Using device: {DEVICE}")

def get_data_loader(data_dir):
    """
    Creates the test data loader with the same transforms used in training.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Standard ImageNet normalization
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Ensure class mapping matches expectation
    print(f"Classes found in dataset: {dataset.classes}")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader

def load_model(model_path, num_classes):
    """
    Loads the EfficientNet-B0 architecture with dropout (matching training setup).
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Initialize EfficientNet-B0 using efficientnet_pytorch (same as training)
    model = EfficientNet.from_pretrained("efficientnet-b0")
    
    # 2. Replace the final layer with dropout + linear (matching training architecture)
    model._fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(model._fc.in_features, num_classes)
    )
    
    # 3. Load the trained state dictionary
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    return model

def generate_confusion_matrix(model, loader):
    all_preds = []
    all_labels = []

    print("Running predictions on Test Set...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_and_save_matrix(y_true, y_pred, classes):
    # Compute matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2) # Adjust font size
    
    # Create Heatmap
    # annot=True shows the numbers, fmt='d' prevents scientific notation
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix: EfficientNet-B0', fontsize=16)
    
    # Save the plot
    output_filename = 'confusion_matrix_result.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion Matrix saved as '{output_filename}'")
    
    # Print Classification Report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_DIR):
        # 1. Prepare Data
        test_loader = get_data_loader(TEST_DIR)
        
        # 2. Load Model
        model = load_model(MODEL_PATH, len(CLASSES))
        
        # 3. Get Predictions
        y_true, y_pred = generate_confusion_matrix(model, test_loader)
        
        # 4. Plot and Report
        plot_and_save_matrix(y_true, y_pred, CLASSES)
    else:
        print("❌ Error: Check your paths!")
        print(f"Looking for Model at: {MODEL_PATH}")
        print(f"Looking for Test Data at: {TEST_DIR}")