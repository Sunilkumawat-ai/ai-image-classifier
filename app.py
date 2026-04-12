import streamlit as st
from PIL import Image
from transformers import pipeline
import os
from datetime import datetime
import re

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🔍",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>🔍 AI Image Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Upload an image and get instant AI prediction</p>",
    unsafe_allow_html=True
)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("image-classification")

classifier = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Predict Image", "Save Training Data"]
)

# ---------------- SMART FILENAME HINT ----------------
def extract_hint(filename):
    name = filename.lower()

    # remove extension
    name = name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")

    # replace separators
    name = name.replace("-", " ").replace("_", " ")

    # check if meaningful (must contain letters)
    if re.search(r"[a-zA-Z]{3,}", name):
        return name
    return None

# ---------------- SAVE DATA FUNCTION ----------------
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

    # ---------------- FILE INFO ----------------
    file_name = uploaded_file.name
    hint = extract_hint(file_name)

    st.write(f"📁 File Name: {file_name}")

    if hint:
        st.info(f"💡 Filename hint (optional): {hint}")
    else:
        st.warning("📁 No meaningful filename detected (ignored)")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- PREDICT MODE ----------------
    if mode == "Predict Image":
        with st.spinner("🤖 AI is thinking..."):
            results = classifier(image)

        top = results[0]
        label = top["label"]
        confidence = top["score"] * 100

        st.success("🎯 Prediction Result")

        st.markdown(f"### 🏷️ {label}")

        st.progress(int(confidence))

        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # ---------------- COMPARISON ----------------
        st.markdown("---")
        st.markdown("### 🧠 AI vs Filename Hint")

        st.write(f"🤖 AI Prediction: {label}")

        if hint:
            st.write(f"📁 Filename Hint: {hint}")
        else:
            st.write("📁 Filename Hint: Not available")

    # ---------------- SAVE MODE ----------------
    else:
        st.info("📌 Add label to save this image for training dataset")

        label_input = st.text_input("Enter label (e.g. cat, dog, tree)")

        if st.button("💾 Save Image"):
            if label_input.strip() == "":
                st.warning("Please enter a label first")
            else:
                path = save_data(image, label_input)
                st.success(f"Saved to: {path}")
