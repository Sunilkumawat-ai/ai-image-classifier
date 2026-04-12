import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Process image
def process_image(image):
    image = image.resize((224,224))
    img = np.array(image)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Predict
def predict(image):
    img = process_image(image)
    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
    return decoded

# UI
st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image and get AI predictions")

file = st.file_uploader("Choose Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")
    result = predict(img)

    for r in result:
        st.write(f"{r[1]} - {round(r[2]*100,2)}%")
