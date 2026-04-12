import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image and AI will predict what it is")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:

    # show image
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction...")

    # compress image (IMPORTANT for avoiding errors)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=60)
    image_bytes = buffer.getvalue()

    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

    headers = {
        "Content-Type": "application/octet-stream"
    }

    result = None

    # retry system (fix errors)
    for i in range(3):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=image_bytes,
                timeout=120
            )
            result = response.json()
            break

        except Exception:
            st.warning(f"Retrying... {i+1}/3")
            time.sleep(2)

    # show result
    if result and isinstance(result, list):

        top = result[0]

        st.success(f"👉 {top['label']} ({round(top['score']*100, 2)}%)")

        st.write("### Top Predictions:")

        for r in result[:5]:
            st.write(f"- {r['label']} : {round(r['score']*100, 2)}%")

    else:
        st.error("AI failed. Try another image or reload app.")
