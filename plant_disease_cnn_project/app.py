"""Streamlit app for plant disease classification."""

from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Classification")
st.write("Upload a plant leaf image to classify its disease category.")

MODEL_PATH = Path("artifacts/best_model.pth")
CLASSES_PATH = Path("artifacts/class_names.json")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("Model files are missing. Run python train.py first.")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)


@st.cache_resource
def load_model():
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(class_names))

    state = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=400)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Classifying image..."):
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0]
                top_probs, top_indices = torch.topk(
                    probabilities, min(3, len(class_names))
                )

        best_index = top_indices[0].item()
        st.subheader("Prediction")
        st.write(f"**Class:** {class_names[best_index]}")
        st.write(f"**Model confidence:** {top_probs[0].item():.2%}")

        st.subheader("Top predictions")
        for probability, index in zip(top_probs, top_indices):
            st.write(
                f"{class_names[index.item()]} — {probability.item():.2%}"
            )

st.caption(
    "Predictions are model estimates, not a definitive plant-health diagnosis."
)
