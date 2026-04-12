import streamlit as st
import requests

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image and AI will tell what it is")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    st.image(file, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")

    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

    try:
        response = requests.post(
            API_URL,
            data=file.getvalue(),
            timeout=60
        )

        result = response.json()

        if isinstance(result, list):
            top = result[0]   # best prediction

            label = top.get("label", "Unknown")
            score = top.get("score", 0)

            st.success(f"👉 {label} ({round(score*100,2)}%)")

            # show top 3 results
            st.write("### Other possibilities:")
            for r in result[:3]:
                st.write(f"- {r['label']} : {round(r['score']*100,2)}%")

        else:
            st.error(result)

    except Exception as e:
        st.error(f"Error: {str(e)}")
