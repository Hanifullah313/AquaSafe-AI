import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
# Import our custom modules 
from dataset import get_dataloaders
from model import WaterQualityNet

# --- 1. Configuration (Hyperparameters) ---
SCALER_SAVE_PATH = r"C:\Users\Hanif ullah laptop\Desktop\AquaSafe AI\models\scaler.pkl"
LEARNING_RATE = 0.001
EPOCHS = 100              
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_PATH = r"C:\Users\Hanif ullah laptop\Desktop\AquaSafe AI\data\preprocessed_data.csv"
MODEL_SAVE_PATH = r"C:\Users\Hanif ullah laptop\Desktop\AquaSafe AI\models\best_model.pth"

def train():
    print(f"🚀 Training on device: {DEVICE}")

    # --- 2. Load Data ---
    print("⏳ Loading data...")
    train_loader, val_loader, scaler = get_dataloaders(CSV_PATH, BATCH_SIZE)

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"✅ Scaler saved to {SCALER_SAVE_PATH}")

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = WaterQualityNet(input_features=9).to(DEVICE)
    
    # Loss Function: BCELoss (Binary Cross Entropy) is standard for 0/1 classification
    criterion = nn.BCELoss()
    
    # Optimizer: Adam is the industry standard for starting new projects
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. The Training Loop ---
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train() # Switch to training mode (enables Dropout)
        running_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            # A. Forward Pass (The Prediction)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # B. Backward Pass (The Learning)
            optimizer.zero_grad()   # Reset gradients
            loss.backward()         # Calculate gradients
            optimizer.step()        # Update weights
            
            running_loss += loss.item()

        # --- 5. Validation Loop (Testing the progress) ---
        model.eval() # Switch to eval mode (disables Dropout)
        correct = 0
        total = 0
        
        with torch.no_grad(): # Don't calculate gradients during validation (saves RAM)
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                
                # Convert probability to 0 or 1 (Threshold 0.5)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Val Accuracy: {accuracy:.4f}")

        # --- 6. Save Best Model (Checkpointing) ---
        # We only save if the current model is better than the previous best.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 Improved! Model saved with accuracy: {accuracy:.4f}")

    print("\n✅ Training Complete. Best Accuracy:", best_accuracy)

if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("../models", exist_ok=True)
    train()