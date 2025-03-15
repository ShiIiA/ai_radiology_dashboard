import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# ---------------------- PAGE CONFIGURATION ----------------------
st.set_page_config(page_title="AI Radiology Risk Dashboard", layout="wide")
st.title("🩻 AI Radiology Risk Dashboard")
st.sidebar.header("🛠 Choose AI Model & Dataset")

# ---------------------- TEMP STORAGE FOR IMAGES ----------------------
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------- MODEL SOURCES ----------------------
model_sources = ["Hugging Face", "GitHub"]
selected_source = st.sidebar.radio("Select Model Source", model_sources)

# Hugging Face Models
hf_models = {
    "ResNet-50 (Medical)": "microsoft/resnet-50",
    "ViT (Vision Transformer)": "google/vit-base-patch16-224",
    "EfficientNet-B3": "timm/efficientnet_b3"
}

# GitHub Models
github_models = {
    "ResNet-18 (GitHub)": "pytorch/vision",
    "MobileNetV2": "pytorch/vision"
}

# ---------------------- MODEL SELECTION ----------------------
if selected_source == "Hugging Face":
    selected_model = st.sidebar.selectbox("Choose a Model", list(hf_models.keys()))
    model_name = hf_models[selected_model]
    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
elif selected_source == "GitHub":
    selected_model = st.sidebar.selectbox("Choose a Model", list(github_models.keys()))
    model_repo = github_models[selected_model]
    model = torch.hub.load(model_repo, model=selected_model.lower(), pretrained=True)
    model.eval()

st.sidebar.success(f"✅ {selected_model} Loaded!")

# ---------------------- IMAGE UPLOAD ----------------------
uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess Image
    inputs = feature_extractor(image, return_tensors="pt")["pixel_values"]

    # Make Prediction
    outputs = model(inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    st.write(f"🔍 **Predicted Class:** {'Disease Detected' if prediction == 1 else 'No Disease'}")

    # ---------------------- SHAP MODEL EXPLAINABILITY ----------------------
    st.header("🔍 AI Model Transparency (SHAP)")
    st.write("Understanding **why** the model made this prediction.")

    explainer = shap.Explainer(model, inputs)
    shap_values = explainer(inputs)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, inputs, show=False)
    st.pyplot(fig)

# ---------------------- DATASET FAIRNESS ANALYSIS ----------------------
st.sidebar.header("⚖️ Dataset Fairness Analysis")
protected = st.sidebar.selectbox("Protected Attribute", ["gender", "age"])

# Simulated dataset (Replace with real medical dataset)
data = {
    "gender": np.random.choice([0, 1], 100),
    "age": np.random.randint(20, 80, 100),
    "disease": np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

# Convert dataset to AI Fairness 360 format
dataset = BinaryLabelDataset(df=df, label_names=["disease"], protected_attribute_names=[protected])
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{protected: 1}], unprivileged_groups=[{protected: 0}])
disparate_impact = metric.disparate_impact()

st.sidebar.write(f"⚖️ **Disparate Impact Ratio:** {disparate_impact:.2f} (Ideal: Close to 1.0)")

# ---------------------- RISK SCORE CALCULATION ----------------------
def calculate_risk_score(misclassification_rate, dataset_bias, explainability_score, fairness_score):
    return round((misclassification_rate * 0.4 + dataset_bias * 0.3 + (1 - explainability_score) * 0.15 + (1 - fairness_score) * 0.15) * 100, 2)

risk_score = calculate_risk_score(1 - accuracy_score(df["disease"], df["disease"]), disparate_impact, 0.8, 0.7)
st.metric(label="AI Risk Score", value=f"{risk_score}/100")

# ---------------------- CLEANUP (DELETE TEMP FILES) ----------------------
import shutil
shutil.rmtree(TEMP_DIR)
