import streamlit as st
from PIL import Image
from transformers import pipeline
import os
import re
from datetime import datetime

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Image Classifier Pro",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Image Classifier Pro")
st.markdown("Upload image → get prediction → give feedback to improve AI")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

classifier = load_model()

# ---------------- FUNCTIONS ----------------
def predict(image):
    result = classifier(image)[0]
    label = result["label"]
    confidence = result["score"] * 100
    return label, confidence


def clean_label(label):
    """Turn user input into a safe folder name (letters, digits, _ and - only).

    Without this, a label like "../../x" could make the app write files
    outside the dataset/ folder (path traversal).
    """
    label = re.sub(r"[^a-z0-9_-]+", "_", label.strip().lower()).strip("_")
    return label[:50]


def save_feedback(image, correct_label):
    folder = os.path.join("dataset", correct_label)
    os.makedirs(folder, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    path = os.path.join(folder, filename)

    # JPEG cannot store transparency, so convert first (PNGs with alpha would crash)
    image.convert("RGB").save(path)
    return path


def confidence_status(conf):
    if conf > 80:
        return "🟢 Very Confident"
    elif conf > 50:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

# ---------------- APP ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # prediction
    with st.spinner("🤖 AI analyzing image..."):
        label, confidence = predict(image)

    st.success(f"🎯 Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.write(confidence_status(confidence))

    # ---------------- FEEDBACK SECTION ----------------
    st.markdown("## 🧠 Feedback System")

    feedback = st.radio(
        "Is prediction correct?",
        ["✔ Yes", "❌ No"]
    )

    if feedback == "❌ No":
        correct_label = st.text_input("Enter correct label (e.g. laptop, dog, cat)")

        if st.button("💾 Save Feedback"):
            safe_label = clean_label(correct_label)
            if safe_label == "":
                st.warning("Please enter a label using letters or numbers")
            else:
                path = save_feedback(image, safe_label)
                st.success(f"Saved for training: {path}")

    else:
        st.success("Great! Thanks for confirming 👍")
