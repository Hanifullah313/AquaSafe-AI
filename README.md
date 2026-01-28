
# 💧 AquaSafe AI: Water Potability Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Leveraging Deep Learning to support UN Sustainable Development Goal 6: Clean Water and Sanitation."**

## 📌 Project Overview
**AquaSafe AI** is an end-to-end Machine Learning application designed to predict whether a water sample is safe for human consumption based on its chemical properties (pH, Sulfate, Chloramines, etc.).

Unlike basic data science notebooks, this project is engineered as a **full-stack AI microservice**:
1.  **Deep Learning Model:** A custom Artificial Neural Network (ANN) built with **PyTorch**.
2.  **API Backend:** A robust REST API serving the model using **FastAPI**.
3.  **Interactive Frontend:** A user-friendly dashboard built with **Streamlit**.
4.  **Explainability:** Integrated Feature Importance analysis to interpret model decisions.

---

## 🚀 Key Features
* **🧠 Custom ANN Architecture:** Engineered a Sequential Neural Network with optimized dropout layers to prevent overfitting on the Water Potability dataset.
* **⚙️ MLOps Architecture:** Decoupled the inference engine (FastAPI) from the user interface (Streamlit) to simulate real-world production environments.
* **📊 Data Engineering Pipeline:** Implemented robust preprocessing including **Mean Imputation** (to retain 100% of data) and **Standard Scaling** for numerical stability.
* **🔍 Explainable AI (XAI):** Implemented **Permutation Importance** to visualize which chemical factors contribute most to the model's predictions.

---

## 🛠️ Tech Stack & Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model Core** | `PyTorch` | Artificial Neural Network (ANN) with Binary Classification |
| **Backend** | `FastAPI` | Asynchronous REST API for model inference |
| **Frontend** | `Streamlit` | Interactive Web UI for real-time predictions |
| **Data Processing** | `Pandas` / `Scikit-Learn` | Data cleaning, imputation, and scaling |
| **Validation** | `Pydantic` | Data schema validation for API requests |

### 🏗️ System Architecture
```mermaid
graph LR
    User[User] -- Input Data --> UI[Streamlit Frontend]
    UI -- JSON Request --> API[FastAPI Backend]
    API -- Tensor --> Model[PyTorch ANN]
    Model -- Prediction --> API
    API -- JSON Response --> UI
    UI -- Visual Result --> User

```

---

## 📂 Project Structure

This project follows industry-standard directory organization:

```
aquasafe-ai/
├── api/
│   └── main.py              # FastAPI application entry point
├── data/
│   └── water_potability.csv # Raw dataset (Kaggle)
├── frontend/
│   └── app.py               # Streamlit dashboard
├── models/
│   ├── best_model.pth       # Trained PyTorch weights
│   └── scaler.pkl           # Saved StandardScaler object
├── src/
│   ├── dataset.py           # Custom PyTorch Dataset & Dataloaders
│   ├── model.py             # ANN Architecture Class
│   ├── train.py             # Training loop with Checkpointing
│   └── explainability.py    # Feature Importance script
├── requirements.txt         # Project dependencies
└── README.md                # Project Documentation

```

---

## ⚡ Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/YourUsername/aquasafe-ai.git](https://github.com/YourUsername/aquasafe-ai.git)
cd aquasafe-ai

```

### 2. Install Dependencies

It is recommended to use a virtual environment (Conda or venv).

```bash
pip install -r requirements.txt

```

### 3. Train the Model (Optional)

The repository includes a pre-trained model. To retrain from scratch:

```bash
cd src
python train.py

```

*This will generate `best_model.pth` and `scaler.pkl` in the `models/` directory.*

### 4. Run the Application

You need to run the Backend and Frontend in separate terminals.

**Terminal 1: Start the Backend (API)**

```bash
cd api
uvicorn main:app --reload

```

*Wait for the message: `Application startup complete*`

**Terminal 2: Start the Frontend (UI)**

```bash
cd frontend
streamlit run app.py

```

*The app will automatically open in your browser at `http://localhost:8501`.*

---

## 📈 Model Performance & Insights

* **Accuracy:** ~70% (Baseline for this complex dataset)
* **Loss Function:** Binary Cross Entropy (BCELoss)
* **Optimizer:** Adam (`lr=0.0001`)

### 🧠 Feature Importance

Using the "Shuffle Test" (Permutation Importance), the model identified **Sulfate** and **pH** as the critical determinants of water safety.

*(Note: If image is missing, run `src/explainability.py` to generate it)*

---

## 🔗 Live Demo

* 
* **Video Demo:** https://www.linkedin.com/posts/hanifullah313_machinelearning-pytorch-fastapi-activity-7422353306825453568-sHqg?utm_source=social_share_send&utm_medium=android_app&rcm=ACoAAFHCdjABH6QTdL3w0tEj_VYHQ4Z3Kr8eDgg&utm_campaign=copy_link


## 👨‍💻 Author

**Hanif Ullah**



### 📝 License

This project is licensed under the MIT License.

