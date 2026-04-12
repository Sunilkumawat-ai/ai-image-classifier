import streamlit as st
from PIL import Image
from transformers import pipeline
import os
from datetime import datetime
import re

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Best AI Image Classifier",
    page_icon="🔍",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>🔍 Best AI Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload image and get smart AI prediction</p>", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

classifier = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Mode", ["Predict", "Save Feedback"])

# ---------------- UTIL FUNCTIONS ----------------
def clean_label(label):
    label = label.lower()

    if "laptop" in label or "notebook" in label:
        return "laptop"
    if "dog" in label:
        return "dog"
    if "cat" in label:
        return "cat"
    if "car" in label:
        return "car"

    return label


def confidence_status(conf):
    if conf > 80:
        return "🟢 Very Confident"
    elif conf > 50:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"


def extract_hint(filename):
    name = filename.lower()
    name = name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
    name = name.replace("-", " ").replace("_", " ")

    if re.search(r"[a-zA-Z]{3,}", name):
        return name
    return None


def save_data(image, label):
    folder = f"dataset/{label}"
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    path = os.path.join(folder, filename)

    image.save(path)
    return path

# ---------------- MAIN APP ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # file info
    file_name = uploaded_file.name
    hint = extract_hint(file_name)

    st.write(f"📁 File: {file_name}")

    if hint:
        st.info(f"💡 Hint: {hint}")
    else:
        st.warning("No useful filename hint")

    st.image(image, use_container_width=True)

    # ---------------- PREDICTION ----------------
    if mode == "Predict":
        with st.spinner("🤖 AI analyzing..."):
            result = classifier(image)[0]

        label = clean_label(result["label"])
        confidence = result["score"] * 100

        st.success(f"🎯 Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.write(confidence_status(confidence))

    # ---------------- FEEDBACK MODE ----------------
    else:
        st.info("Help improve AI by giving correct label")

        label_input = st.text_input("Enter correct label")

        if st.button("Save for training"):
            if label_input.strip() == "":
                st.warning("Please enter label")
            else:
                path = save_data(image, label_input)
                st.success(f"Saved: {path}")
