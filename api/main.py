from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os
import joblib
import pandas as pd

# --- 1. Setup Paths ---
# We need to tell Python where to find the 'src' folder to import the model
# This looks up one level (..) and finds 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import WaterQualityNet

# --- 2. Configuration ---
SCALER_PATH = "../models/scaler.pkl"
MODEL_PATH = "../models/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. Define the Input Data Format (Schema) ---
# This ensures the API only accepts valid numbers
class WaterData(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

# --- 4. Initialize API & Load Model ---
app = FastAPI(title="AquaSafe AI API")


print(f"⏳ Loading model from {MODEL_PATH} on {DEVICE}...")



try:
    # Initialize the architecture (Must match your src/model.py)
    model = WaterQualityNet(input_features=9)
    scaler = None # Initialize variable
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # CRITICAL: Turn off Dropout for prediction
    print("✅ Model loaded successfully!")

    scaler = joblib.load(SCALER_PATH) # <--- Load the tool
    print("✅ Model and Scaler loaded successfully!")

except FileNotFoundError:
    print("❌ ERROR: 'best_model.pth' not found. Did you run 'src/train.py'?")
    # We don't crash the app, but predictions will fail if requested
    model = None
except Exception as e:
    print(f"❌ ERROR: Could not load model. Details: {e}")
    model = None

# --- 5. The Prediction Endpoint ---
@app.post("/predict")
def predict_water_quality(data: WaterData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="System not ready")

    # 1. Create a list of the raw inputs
    raw_data = [[
        data.ph, data.hardness, data.solids, data.chloramines,
        data.sulfate, data.conductivity, data.organic_carbon,
        data.trihalomethanes, data.turbidity
    ]]
    
    # 2. SCALE THE DATA (The Missing Step!)
    # We transform the raw numbers (e.g. 20000) into scaled numbers (e.g. 0.5)
    # Note: We use pandas DataFrame to avoid warnings, or just pass list if version allows
    scaled_data = scaler.transform(raw_data)
    
    # 3. Convert to Tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(DEVICE)
    
    # 4. Predict
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
    
    return {
        "prediction": "Safe to Drink" if probability > 0.5 else "Not Safe",
        "confidence_score": round(probability, 4)
    }

# --- 6. The Root Endpoint (Just to check if it's working) ---
@app.get("/")
def home():
    return {"message": "AquaSafe AI API is running!"}