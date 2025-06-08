import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# Streamlit page config
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🧴 Skin Cancer Classification")
st.write("Upload a skin lesion image to predict the type of skin cancer using a deep learning model.")

# File ID from Google Drive shareable link
drive_file_id = "YOUR_FILE_ID_HERE"  # 🔁 Replace this with your real Google Drive file ID
model_path = "best_model_vgg16.keras"

# Download model if not already present
if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define your 9 classes
class_names = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Melanoma",
    "Vascular lesions",
    "Squamous cell carcinoma",
    "Seborrheic keratoses"
]

# Upload image
uploaded_file = st.file_uploader("📤 Upload a skin lesion image", type=["jpg", "jpeg", "png"])

# When image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((128, 128))  # Match model input size
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.success(f"🔍 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}**")

else:
    st.warning("Please upload an image to begin.")
