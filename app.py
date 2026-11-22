import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification

st.set_page_config(page_title="KisanNetra AI | Multi-Crop XAI Diagnostics", layout="wide")

@st.cache_resource
def load_model():
    model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess, model

preprocess, model = load_model()

def generate_gradcam(img_pil, model, preprocess):
    input_tensor = preprocess(img_pil).unsqueeze(0)
    
    activations = []
    gradients = []
    
    # Hook into final convolutional block of MobileNetV2
    target_layer = model.mobilenet_v2.layer[-1]
    
    def forward_hook(module, inp, out):
        activations.append(out)
        
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    outputs = model(input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100
    raw_label = model.config.id2label[pred_idx]
    
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
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    orig_resized = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * orig_resized + 0.45 * heatmap)
    
    return raw_label, confidence, orig_resized, heatmap, overlay

def get_kvk_advisory(label):
    low_label = label.lower()
    if "healthy" in low_label:
        return (
            "Healthy Foliage (No Pathogen Detected)",
            "Maintain standard irrigation and balanced NPK nutrition. Preventive spray of Cold-Pressed Neem Oil (1500 ppm @ 3 ml/L) every 15 days.",
            "No chemical fungicide or bactericide required."
        )
    elif any(k in low_label for k in ["blight", "mildew", "spot", "rust", "scab", "rot"]):
        return (
            "Fungal / Oomycete Foliar Pathology",
            "Foliar spray of Trichoderma viride (5 g/L) or Copper Oxychloride 50 WP (2.5 g/L) during early lesion onset.",
            "Systemic curative spray of Mancozeb 75 WP (2.0 g/L) or Azoxystrobin 23% SC (1.0 ml/L) at 10-day intervals."
        )
    elif any(k in low_label for k in ["virus", "mosaic", "curl"]):
        return (
            "Viral Vector-Borne Pathology (Whitefly/Aphid Transmitted)",
            "Install Yellow Sticky Traps (15/acre) and spray Neem Seed Kernel Extract (NSKE 5%). Rogue out severely infected plants.",
            "Control sap-sucking vectors using Imidacloprid 17.8% SL (0.5 ml/L) or Thiamethoxam 25% WG (0.3 g/L)."
        )
    elif "bacterial" in low_label:
        return (
            "Bacterial Foliar Infection",
            "Prune infected foliage and apply Pseudomonas fluorescens (5 g/L) foliar bio-bactericide.",
            "Spray Copper Hydroxide 77% WP (2.0 g/L) + Streptocycline (0.15 g/L) per KVK guidelines."
        )
    else:
        return (
            "Pest / Mite Foliar Infestation",
            "Release predatory mites or apply Horticultural Mineral Oil / Neem Oil (5 ml/L).",
            "Targeted acaricide application: Spiromesifen 22.9% SC (0.8 ml/L) on leaf undersides."
        )

st.markdown("<h1 style='text-align: center; color: #1b5e20;'>KisanNetra AI — Explainable Multi-Crop Pathology System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Universal 38-Class Coverage (14 Crop Species) | Real-Time Grad-CAM Verification | KVK Agronomic Advisories</p>", unsafe_allow_html=True)

with st.expander("View Supported Crops & Pathologies (38 Classes across 14 Species)"):
    st.write("**Supported Crops:** Tomato, Potato, Bell Pepper, Corn (Maize), Apple, Grape, Orange, Peach, Cherry, Strawberry, Soybean, Squash, Blueberry, Raspberry.")
    st.info("Note: For accurate Grad-CAM lesion localization, upload clear close-up images of individual leaves against a plain or natural background.")

uploaded_file = st.file_uploader("Upload Crop Leaf Image (JPG, JPEG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Running MobileNetV2 Inference & Computing Grad-CAM Gradients..."):
            label, conf, orig_img, heatmap_img, overlay_img = generate_gradcam(img, model, preprocess)
            
        st.markdown("---")
        st.success(f"### Diagnosis: **{label}** | Confidence Score: **{conf:.2f}%**")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(orig_img, caption="1. Standardized Input (224×224)", use_container_width=True)
        with c2:
            st.image(heatmap_img, caption="2. Grad-CAM Saliency Map", use_container_width=True)
        with c3:
            st.image(overlay_img, caption="3. Diagnostic Lesion Overlay", use_container_width=True)
            
        pathology_type, organic_rx, chem_rx = get_kvk_advisory(label)
        st.markdown("---")
        st.subheader(f"KVK-Aligned Agronomic Protocol ({pathology_type})")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Tier-1: Bio-Control & Organic Management**\n\n{organic_rx}")
        with col_b:
            st.warning(f"**Tier-2: Targeted Chemical Intervention (KVK Standard)**\n\n{chem_rx}")
    except Exception as e:
        st.error(f"Error processing image: {e}. Please upload a valid JPG or PNG leaf image.")
