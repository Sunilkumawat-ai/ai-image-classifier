import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load labels
import requests
labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

st.title("🧠 Image Recognition AI")
st.write("Upload an image and AI will predict what it is")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = labels[predicted.item()]

    st.subheader("Prediction:")
    st.success(label)
