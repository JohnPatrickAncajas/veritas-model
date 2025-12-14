import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from efficientnet_pytorch import EfficientNet

# ================= CONFIGURATION =================
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

print(f"Using device: {DEVICE}")

def get_data_loader(data_dir):
    """Creates the test data loader."""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print(f"Classes found in dataset: {dataset.classes}")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset

def load_model(model_path, num_classes):
    """Loads the EfficientNet-B0 model."""
    print(f"Loading model from: {model_path}")
    
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(model._fc.in_features, num_classes)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def calculate_per_class_accuracy(model, loader, dataset):
    """Calculate accuracy for each class."""
    class_correct = {i: 0 for i in range(len(CLASSES))}
    class_total = {i: 0 for i in range(len(CLASSES))}
    
    print("Running predictions on Test Set...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Count correct predictions per class
            for label, pred in zip(labels, preds):
                label_idx = label.item()
                class_total[label_idx] += 1
                if label == pred:
                    class_correct[label_idx] += 1
    
    # Calculate percentages
    class_accuracies = {}
    for i in range(len(CLASSES)):
        if class_total[i] > 0:
            accuracy = (class_correct[i] / class_total[i]) * 100
            class_accuracies[CLASSES[i]] = accuracy
            print(f"{CLASSES[i].upper()}: {class_correct[i]}/{class_total[i]} ({accuracy:.1f}%)")
        else:
            class_accuracies[CLASSES[i]] = 0.0
    
    return class_accuracies

def plot_class_accuracy(class_accuracies):
    """Create and save the bar chart."""
    # Define nice colors for each class (matching the image style)
    colors = ['#4C5C7E', '#3B7B7D', '#3A9D8F', '#7CB342']
    
    # Prepare data
    categories = list(class_accuracies.keys())
    # Map to display names
    display_names = {
        '2d': '2D Art',
        '3d': '3D Art', 
        'ai': 'AI-Generated',
        'real': 'Real Human'
    }
    x_labels = [display_names.get(cat, cat) for cat in categories]
    accuracies = [class_accuracies[cat] for cat in categories]
    
    # Calculate average
    avg_accuracy = np.mean(accuracies)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_labels, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on top of each bar
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Formatting
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.title(f'Classification Accuracy per Category (Avg: ~{avg_accuracy:.1f}%)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Make it look clean
    plt.tight_layout()
    
    # Save
    output_filename = 'class_accuracy_chart.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Class accuracy chart saved as '{output_filename}'")
    print(f"📊 Average Accuracy: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_DIR):
        # 1. Prepare Data
        test_loader, test_dataset = get_data_loader(TEST_DIR)
        
        # 2. Load Model
        model = load_model(MODEL_PATH, len(CLASSES))
        
        # 3. Calculate Per-Class Accuracy
        class_accuracies = calculate_per_class_accuracy(model, test_loader, test_dataset)
        
        # 4. Create and Save Bar Chart
        plot_class_accuracy(class_accuracies)
    else:
        print("❌ Error: Check your paths!")
        print(f"Looking for Model at: {MODEL_PATH}")
        print(f"Looking for Test Data at: {TEST_DIR}")
