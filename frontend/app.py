import streamlit as st
import requests
import json
import os
from PIL import Image

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"
# Path to the feature importance image (Relative to the frontend folder)
IMPORTANCE_IMAGE_PATH = os.path.join("..", "models", "feature_importance.png")

st.set_page_config(page_title="AquaSafe AI", page_icon="💧", layout="centered")

# --- Header ---
st.title("💧 AquaSafe AI")
st.markdown("**Water Potability Prediction System** | *Powered by PyTorch & FastAPI*")
st.divider()

# --- Input Form ---
col1, col2, col3 = st.columns(3)
with col1:
    ph = st.number_input("pH Level", 0.0, 14.0, 7.0, 0.1)
    hardness = st.number_input("Hardness", 0.0, value=200.0)
    solids = st.number_input("Solids (ppm)", 0.0, value=20000.0)
with col2:
    chloramines = st.number_input("Chloramines", 0.0, value=7.0)
    sulfate = st.number_input("Sulfate", 0.0, value=300.0)
    conductivity = st.number_input("Conductivity", 0.0, value=400.0)
with col3:
    organic_carbon = st.number_input("Organic Carbon", 0.0, value=15.0)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, value=60.0)
    turbidity = st.number_input("Turbidity", 0.0, value=4.0)

# --- Prediction Logic ---
st.divider()
if st.button("🔍 Analyze Water Quality", type="primary"):
    payload = {
        "ph": ph, "hardness": hardness, "solids": solids,
        "chloramines": chloramines, "sulfate": sulfate, "conductivity": conductivity,
        "organic_carbon": organic_carbon, "trihalomethanes": trihalomethanes, "turbidity": turbidity
    }
    
    try:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            pred = result["prediction"]
            conf = result["confidence_score"]
            
            if pred == "Safe to Drink":
                st.success(f"✅ Result: Safe (Confidence: {conf*100:.1f}%)")
                st.balloons()
            else:
                st.error(f"⚠️ Result: Not Safe (Confidence: {conf*100:.1f}%)")
        else:
            st.error("Error: Could not get prediction.")
    except requests.exceptions.ConnectionError:
        st.error("❌ Error: Backend is not running.")

# --- NEW SECTION: Model Explainability ---
st.divider()
st.subheader("🧠 Model Insights")

# We use an expander so it doesn't clutter the screen
with st.expander("How does the AI decide? (Click to View)"):
    st.write("""
    This model uses **Permutation Importance** to determine which chemical factors 
    impact water safety the most.
    """)
    
    # Check if the image exists before trying to open it
    if os.path.exists(IMPORTANCE_IMAGE_PATH):
        image = Image.open(IMPORTANCE_IMAGE_PATH)
        st.image(image, caption="Global Feature Importance (The 'Sabotage Test' Results)", use_container_width=True)
        st.info("💡 **Interpretation:** The higher the bar, the more 'critical' that chemical is for the prediction.")
    else:
        st.warning("⚠️ Feature Importance graph not found. Please run `src/explainability.py` first.")