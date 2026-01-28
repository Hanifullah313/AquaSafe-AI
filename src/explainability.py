import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataset import get_dataloaders
from model import WaterQualityNet

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\Users\Hanif ullah laptop\Desktop\AquaSafe AI\models\best_model.pth"
CSV_PATH = r"C:\Users\Hanif ullah laptop\Desktop\AquaSafe AI\data\preprocessed_data.csv"
FEATURE_NAMES = [
    "pH", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
]

def calculate_permutation_importance(model, dataloader):
    model.eval()
    original_accuracy = 0.0
    total = 0
    correct = 0
    
    # 1. Calculate Baseline Accuracy (Original)
    # We collect all data first to make shuffling easier
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            all_features.append(features)
            all_labels.append(labels)
            
    # Combine batches into one big tensor
    X = torch.cat(all_features).to(DEVICE)
    y = torch.cat(all_labels).to(DEVICE)
    
    # Get Baseline Score
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    baseline_acc = (predicted == y).sum().item() / y.size(0)
    print(f"✅ Baseline Accuracy: {baseline_acc:.4f}")

    importances = {}

    # 2. Loop through each feature and shuffle it
    for i, feature_name in enumerate(FEATURE_NAMES):
        # Create a copy of input data so we don't break the original X
        X_shuffled = X.clone()
        
        # Shuffle ONLY column 'i'
        # We generate random indices to mix up the rows for this specific column
        idx = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, i] = X_shuffled[idx, i]
        
        # Predict again with the "broken" feature
        outputs = model(X_shuffled)
        predicted = (outputs > 0.5).float()
        shuffled_acc = (predicted == y).sum().item() / y.size(0)
        
        # Importance = How much accuracy did we LOSE?
        # If baseline is 0.70 and shuffled is 0.50, importance is 0.20
        drop = baseline_acc - shuffled_acc
        importances[feature_name] = drop
        print(f"   📉 Shuffling {feature_name:<15} -> New Acc: {shuffled_acc:.4f} (Drop: {drop:.4f})")

    return importances

def plot_importance(importances):
    # Sort features by importance
    sorted_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    names = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    # Create Plot
    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color='#4CAF50')
    plt.xlabel('Drop in Accuracy (Importance)')
    plt.title('Feature Importance (Permutation Method)')
    plt.gca().invert_yaxis() # Highest importance on top
    plt.savefig("../models/feature_importance.png")
    print("📊 Graph saved to models/feature_importance.png")
    plt.show()

if __name__ == "__main__":
    # Load Model
    model = WaterQualityNet(input_features=9).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Load Validation Data (We test importance on data the model hasn't memorized)
    _, val_loader = get_dataloaders(CSV_PATH, batch_size=32)
    
    # Run Analysis
    print("🔍 Calculating Feature Importance...")
    scores = calculate_permutation_importance(model, val_loader)
    plot_importance(scores)