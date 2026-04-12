import streamlit as st
from PIL import Image
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Image Classifier Pro",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Image Classifier Pro")
st.markdown("Upload an image and get AI prediction")

# ---------------- MODEL (STABLE - NO CRASH) ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

classifier = load_model()

# ---------------- PREDICTION ----------------
def predict(image):
    results = classifier(image)
    top = results[0]
    label = top["label"]
    confidence = top["score"] * 100
    return label, confidence

# ---------------- CONFIDENCE STATUS ----------------
def confidence_status(conf):
    if conf > 80:
        return "🟢 Very Confident"
    elif conf > 50:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

# ---------------- MAIN APP ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    with st.spinner("🤖 AI analyzing image..."):
        label, confidence = predict(image)

    st.success(f"🎯 Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.write(confidence_status(confidence))
