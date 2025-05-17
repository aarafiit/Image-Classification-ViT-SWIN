import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, SwinForImageClassification, AutoConfig
from safetensors.torch import load_file

# Constants
NUM_CLASSES = 15
CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "No finding"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load models from config + safetensors
@st.cache_resource
def load_models():
    # ViT
    vit_config = AutoConfig.from_pretrained("vit-model")
    vit_config.problem_type = "multi_label_classification"
    vit_model = ViTForImageClassification(vit_config)
    vit_model.load_state_dict(load_file("vit-model/model.safetensors"))
    vit_model.eval()

    # Swin
    swin_config = AutoConfig.from_pretrained("swin-model")
    swin_config.problem_type = "multi_label_classification"
    swin_model = SwinForImageClassification(swin_config)
    swin_model.load_state_dict(load_file("swin-model/model.safetensors"))
    swin_model.eval()

    return vit_model, swin_model

vit_model, swin_model = load_models()

# Streamlit UI
st.title("Chest X-ray Abnormalities Detection")
st.markdown("Upload a chest X-ray image in JPG format and choose a model to detect abnormalities.")

model_choice = st.radio("Select model:", ("ViT", "Swin Transformer"))

uploaded_file = st.file_uploader("Choose a JPG image", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        if model_choice == "ViT":
            logits = vit_model(input_tensor).logits
        else:
            logits = swin_model(input_tensor).logits

        probs = torch.sigmoid(logits).squeeze()
        preds = [CLASS_NAMES[i] for i, p in enumerate(probs) if p > 0.5]

    st.subheader(f"{model_choice} Predictions:")
    st.write(preds if preds else ["No Finding"])
