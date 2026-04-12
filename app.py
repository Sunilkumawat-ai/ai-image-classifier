import streamlit as st
from PIL import Image
from transformers import pipeline

# Set up the page UI
st.set_page_config(page_title="Universal Image AI", page_icon="🔍")

st.title("🌍 Universal Image Identifier")
st.markdown("Upload any image, and the AI will describe what it sees in detail.")

# Load the image-to-text model
# We use st.cache_resource so the model only loads once into memory
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

captioner = load_model()

# Image uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Create a button to trigger analysis
    if st.button("Identify Image"):
        with st.spinner('Thinking...'):
            try:
                # Generate description
                results = captioner(image)
                description = results[0]['generated_text']
                
                # Display results
                st.success("Analysis Complete!")
                st.subheader("What I see:")
                st.write(description.capitalize())
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()
st.caption("Powered by Hugging Face Transformers & Streamlit")
