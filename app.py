import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("🔍 Image Classification AI")

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def predict(image):
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)

    result = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0]

    label = result[1]
    confidence = result[2] * 100

    return label, confidence


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    label, confidence = predict(image)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
