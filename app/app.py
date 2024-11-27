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
    This model uses **CNN** and **Logistic Regression** to make the predictions.
""")

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        # Convert image to grayscale if it's RGB
        if image.mode == "RGB":
            image = image.convert("L")
        
        # Resize the image to 150x150 (input size for the models)
        image = image.resize((150, 150))
        
        # Convert image to a numpy array and normalize pixel values
        img_array = np.array(image) / 255.0

        # Reshape for CNN (add batch and channel dimensions)
        img_cnn = img_array.reshape(1, 150, 150, 1)

        # Flatten the image for Logistic Regression (22500 features)
        img_flat = img_array.flatten().reshape(1, -1)

        return img_cnn, img_flat
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None

# Upload the image
file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if file is not None:
    # Load and display the image
    img = Image.open(file)
    st.image(img, caption="Uploaded X-Ray Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img_cnn, img_flat = preprocess_image(img)
    
    if img_cnn is not None and img_flat is not None:
        # CNN Model Prediction
        cnn_preds = cnn_model.predict(img_cnn)
        cnn_preds = (cnn_preds > 0.5).astype(int)

        # Logistic Regression Model Prediction
        lr_preds = lr_model.predict(img_flat)

        # Display Results
        st.subheader("Prediction Results")
        st.write(f"CNN Prediction: {'Pneumonia' if cnn_preds == 0 else 'Normal'}")
        st.write(f"Logistic Regression Prediction: {'Pneumonia' if lr_preds == 0 else 'Normal'}")
    else:
        st.error("Image preprocessing failed. Please try again.")
