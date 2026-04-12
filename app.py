import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# ---------------- UI ----------------
st.set_page_config(page_title="AI Object Detector", layout="centered")

st.title("🔍 AI Object Detection (YOLOv8)")
st.markdown("Upload image and detect real objects")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # pretrained model

model = load_model()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model(image)

    result = results[0]

    # plot image with boxes
    plotted = result.plot()

    st.image(plotted, caption="Detected Objects")

    st.success("Detection complete ✔️")
