import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from transformers import AutoModelForImageClassification

st.set_page_config(
    page_title="KisanNetra — फसल स्वास्थ्य जांच",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Tiro+Devanagari+Hindi&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #1F2A1E;
    }
    .stApp {
        background-color: #EDF2EB;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 1100px;
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
    }

    .farm-header {
        background-color: #1E4620;
        border-bottom: 6px solid #D97706;
        border-radius: 14px;
        padding: 1.8rem 2.2rem;
        color: #FFFDF7;
        margin-bottom: 1.6rem;
    }
    .hindi-greet {
        font-family: 'Tiro Devanagari Hindi', serif;
        font-size: 1.15rem;
        color: #FCD34D;
        margin-bottom: 0.25rem;
    }
    .farm-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
    }
    .farm-tagline {
        font-size: 0.98rem;
        color: #E2ECE1;
        margin-top: 0.45rem;
        max-width: 720px;
    }

    .step-card {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-top: 4px solid #2E6F40;
        border-radius: 12px;
        padding: 1.25rem;
        height: 100%;
    }
    .step-num {
        display: inline-block;
        background-color: #2E6F40;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 2px 10px;
        border-radius: 6px;
        margin-bottom: 0.55rem;
    }
    .step-heading {
        font-size: 1.02rem;
        font-weight: 600;
        color: #1E3F20;
        margin-bottom: 0.35rem;
    }
    .step-desc {
        font-size: 0.88rem;
        color: #4A5849;
        line-height: 1.5;
        margin: 0;
    }

    .result-banner {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-top: 0.8rem;
        margin-bottom: 1.1rem;
        height: 100%;
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #5C6B5B;
    }
    .result-disease {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1E3F20;
        margin: 0.25rem 0 0.5rem 0;
        line-height: 1.25;
    }
    .badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-sick {
        background-color: #FFEDD5;
        color: #9A3412;
        border: 1px solid #FDBA74;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }

    .remedy-organic {
        background-color: #F3FAF1;
        border: 1.5px solid #B7D9B1;
        border-left: 6px solid #2E6F40;
        border-radius: 12px;
        padding: 1.35rem;
        height: 100%;
    }
    .remedy-chemical {
        background-color: #FFF9EB;
        border: 1.5px solid #F3D59B;
        border-left: 6px solid #B8502A;
        border-radius: 12px;
        padding: 1.35rem;
        height: 100%;
    }
    .remedy-head {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .remedy-text {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #263325;
        margin: 0;
    }

    .crop-box {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-radius: 10px;
        padding: 0.95rem 1.25rem;
        margin-top: 1.4rem;
        font-size: 0.88rem;
        color: #2D3B2C;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    # Standard MobileNetV2 [-1, 1] normalization matching Hugging Face MobileNetV2FeatureExtractor
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return preprocess, model

preprocess, model = load_model()

def format_disease_name(raw_label):
    cleaned = raw_label.replace("___", " — ").replace("__", " ").replace("_", " ")
    return " ".join(cleaned.split())

def generate_gradcam_and_severity(img_pil, model, preprocess, selected_crop="Auto-Detect (All 14 Crops)"):
    input_tensor = preprocess(img_pil).unsqueeze(0)
    activations, gradients = [], []
    target_layer = model.mobilenet_v2.layer[-1]
    
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    
    outputs = model(input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]

    # If farmer selected their specific crop, mask probabilities to that crop's disease classes
    if selected_crop != "Auto-Detect (All 14 Crops)":
        crop_key = selected_crop.split(" ")[0].lower()
        masked_probs = torch.zeros_like(probs)
        for idx, lbl in model.config.id2label.items():
            if crop_key in lbl.lower():
                masked_probs[idx] = probs[idx]
        if masked_probs.sum() > 0:
            probs = masked_probs / masked_probs.sum()

    k = min(3, torch.count_nonzero(probs).item())
    top3_probs, top3_indices = torch.topk(probs, max(k, 1))
    pred_idx = top3_indices[0].item()
    confidence = top3_probs[0].item() * 100
    raw_label = model.config.id2label[pred_idx]
    
    top3_list = [
        (format_disease_name(model.config.id2label[idx.item()]), prob.item() * 100)
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    
    model.zero_grad()
    logits[0, pred_idx].backward()
    fh.remove()
    bh.remove()

    act = activations[0].detach().numpy()[0]
    grad = gradients[0].detach().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
