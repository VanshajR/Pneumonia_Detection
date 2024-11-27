import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# Load models (Make sure you have saved them as discussed earlier)
with open('models/logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)


cnn_model = load_model('models/cnn_model.h5')

# Streamlit UI
st.title("Chest X-Ray Pneumonia Detection")

st.markdown("""
    **Upload an X-Ray image** and this app will predict whether the image shows **Pneumonia** or **Normal**. 
    This model uses **CNN**, **Logistic Regression**, **SVM**, and **XGBoost** to make the predictions.
    """)

# Upload the image
file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Load and display image
    img = Image.open(file)
    st.image(img, caption="Uploaded X-Ray Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for the models
    img_array = np.array(img.resize((150, 150)))  # Resize to match the model's input size
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # CNN Model Prediction
    cnn_preds = cnn_model.predict(img_array)
    cnn_preds = (cnn_preds > 0.5).astype(int)

    # Logistic Regression Model Prediction (no PCA, just flatten the image)
    img_flat = img_array.flatten().reshape(1, -1)  # Flatten the image to 22500 features
    lr_preds = lr_model.predict(img_flat)


    # Display Results
    st.subheader("Prediction Results")
    st.write(f"CNN Prediction: {'Pneumonia' if cnn_preds == 0 else 'Normal'}")
    st.write(f"Logistic Regression Prediction: {'Pneumonia' if lr_preds == 0 else 'Normal'}")
