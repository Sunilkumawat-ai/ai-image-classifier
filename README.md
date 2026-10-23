# 🔍 AI Image Classifier

A Streamlit web app that recognizes what is in a photo using a pretrained **Vision Transformer (ViT)** from Hugging Face, and lets users send back corrections to build a small labeled dataset.



![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)




![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)




![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?logo=pytorch&logoColor=white)




![License](https://img.shields.io/badge/License-MIT-green)



🔗 **Live demo:** [Open the app](https://sunil-ai-image-classifier-jnnscyx2gm87sbfbhsygp7.streamlit.app/) (free hosting, so the first open can take a minute to wake up)

## ✨ Features

- Upload a JPG or PNG image
- Get the top prediction with a confidence score
- Confidence status: 🟢 above 80%, 🟡 above 50%, 🔴 50% or below
- Feedback loop: if the prediction is wrong, enter the correct label and the image is saved to `dataset/<label>/`

## 🧠 How it works

1. The image is passed to the Hugging Face `image-classification` pipeline.
2. The model is [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224), pretrained on ImageNet-21k and fine-tuned on ImageNet-1k, so it predicts one of the 1,000 ImageNet classes.
3. The app shows the highest-scoring class and its confidence.
4. If the user marks it wrong, the corrected label is cleaned and the image is stored for later use.

## ⚠️ Limitations

- It only knows the 1,000 ImageNet classes. Anything outside them gets the closest wrong guess.
- Feedback only **collects** data. The model is not retrained automatically.
- Saved feedback lives on the local disk, so it can be lost on hosts that reset their storage.
- Accuracy has not been benchmarked in this project.

## 🚀 Run locally

```bash
git clone https://github.com/Sunilkumawat-ai/ai-image-classifier.git
cd ai-image-classifier
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights, so it needs internet and takes a little longer. A Python 3.11 dev container is also included for GitHub Codespaces.

## 🔒 Security notes

- Feedback labels are sanitized to letters, digits, `_` and `-` (max 50 characters) before they are used as folder names, which prevents path traversal such as `../../x`.
- Images are converted to RGB and re-encoded as JPEG before saving.
- The `dataset/` folder is git-ignored so user uploads are never committed.

## 📁 Project structure

```
ai-image-classifier/
├── app.py              # Streamlit app: UI, prediction, feedback
├── requirements.txt    # Python dependencies
├── .devcontainer/      # Codespaces / VS Code dev container
├── .gitignore
├── LICENSE
└── README.md
```

## 🗺️ Roadmap

- [ ] Show the top 3 predictions
- [ ] Measure accuracy on a small test set
- [ ] Test robustness against adversarial images and add a defense
- [ ] Review step before feedback images are added to the dataset

## 📄 License

MIT. See [LICENSE](LICENSE).

## 👨‍💻 Author

**Sunil Kumawat**, building toward AI security engineering.
[GitHub](https://github.com/Sunilkumawat-ai) · [LinkedIn](https://www.linkedin.com/in/sunil-kumawat-ai)
