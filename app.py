import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification

st.set_page_config(
    page_title="KisanNetra — Crop Pathology Diagnostics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean editorial aesthetic (warm stone background, crisp borders, zero gradient fluff)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #18181B;
    }
    .stApp {
        background-color: #FAF9F6;
    }
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Editorial Masthead */
    .masthead {
        border-bottom: 1px solid #E4E4E7;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .brand-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #14281D;
        line-height: 1.1;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #52525B;
        margin-top: 0.35rem;
    }
    .meta-tag {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #3F6212;
        background: #ECFCCB;
        border: 1px solid #D9F99D;
        padding: 4px 10px;
        border-radius: 4px;
    }

    /* Structured Cards */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 8px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-kicker {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #71717A;
        margin-bottom: 0.4rem;
    }
    .diagnosis-heading {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #18181B;
        margin: 0;
        line-height: 1.25;
    }
    .status-pill-healthy {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 99px;
        margin-top: 0.6rem;
    }
    .status-pill-alert {
        display: inline-block;
        background: #FEE2E2;
        color: #991B1B;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 99px;
        margin-top: 0.6rem;
    }

    /* Treatment Grid */
    .protocol-box {
        background: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 8px;
        padding: 1.35rem;
        height: 100%;
    }
    .protocol-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #18181B;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F4F4F5;
    }
    .protocol-body {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #3F3F46;
        margin: 0;
    }

    /* Crop List Strip */
    .crop-strip {
        font-size: 0.84rem;
        color: #52525B;
        background: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

def format_disease_name(raw_label):
    # Clean up underscores/triple underscores from dataset labels into readable titles
    cleaned = raw_label.replace("___", " — ").replace("__", " ").replace("_", " ")
    return " ".join(cleaned.split())

def generate_gradcam(img_pil, model, preprocess):
    input_tensor = preprocess(img_pil).unsqueeze(0)
    activations, gradients = [], []
    target_layer = model.mobilenet_v2.layer[-1]
    
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    
    outputs = model(input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]
    
    top3_probs, top3_indices = torch.topk(probs, 3)
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
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    orig_resized = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * orig_resized + 0.45 * heatmap)
    
    return raw_label, format_disease_name(raw_label), confidence, top3_list, orig_resized, heatmap, overlay

def get_kvk_advisory(label):
    low_label = label.lower()
    if "healthy" in low_label:
        return (
            "Healthy Specimen",
            "Maintain standard irrigation and balanced NPK nutrition. Apply a preventive foliar spray of Cold-Pressed Neem Oil (1500 ppm @ 3 ml/L) at 15-day intervals.",
            "No chemical fungicide or bactericide intervention is warranted."
        )
    elif any(k in low_label for k in ["blight", "mildew", "spot", "rust", "scab", "rot"]):
        return (
            "Fungal / Oomycete Pathology",
            "Apply a foliar spray of Trichoderma viride (5 g/L) or Copper Oxychloride 50 WP (2.5 g/L) at initial lesion onset.",
            "Administer a systemic curative spray of Mancozeb 75 WP (2.0 g/L) or Azoxystrobin 23% SC (1.0 ml/L) every 10 days."
        )
    elif any(k in low_label for k in ["virus", "mosaic", "curl"]):
        return (
            "Viral Vector-Borne Pathology",
            "Install Yellow Sticky Traps (15/acre) and spray Neem Seed Kernel Extract (NSKE 5%). Rogue out severely symptomatic plants.",
            "Control whitefly/aphid vectors using Imidacloprid 17.8% SL (0.5 ml/L) or Thiamethoxam 25% WG (0.3 g/L)."
        )
    elif "bacterial" in low_label:
        return (
            "Bacterial Foliar Infection",
            "Prune symptomatic foliage and apply Pseudomonas fluorescens (5 g/L) foliar bio-bactericide.",
            "Spray Copper Hydroxide 77% WP (2.0 g/L) combined with Streptocycline (0.15 g/L) per KVK schedule."
        )
    else:
        return (
            "Pest / Mite Infestation",
            "Release predatory mites or apply Horticultural Mineral Oil / Neem Oil (5 ml/L) targeting leaf undersides.",
            "Apply targeted acaricide: Spiromesifen 22.9% SC (0.8 ml/L) under field supervision."
        )

# Masthead
st.markdown("""
<div class="masthead">
    <div>
        <h1 class="brand-title">KisanNetra</h1>
        <div class="brand-sub">Explainable Multi-Crop Pathology & Agronomic Decision Support</div>
    </div>
    <div class="meta-tag">38 Classes · 14 Species · Grad-CAM XAI</div>
</div>
""", unsafe_allow_html=True)

# Upload Zone
uploaded_file = st.file_uploader(
    "Upload a high-resolution leaf specimen (JPG or PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is None:
    st.markdown("""
    <div class="clean-card" style="margin-top: 1rem;">
        <div class="card-kicker">Diagnostic Pipeline</div>
        <p style="color: #3F3F46; font-size: 0.95rem; line-height: 1.65; margin: 0;">
            Upload an individual leaf photograph to run real-time classification across 38 plant pathologies. 
            The system extracts spatial gradients from the terminal convolutional block of MobileNetV2 to generate a 
            pixel-level Grad-CAM saliency map alongside dual-tier Krishi Vigyan Kendra (KVK) field recommendations.
        </p>
    </div>
    <div class="crop-strip">
        <strong>Supported Coverage (14 Crops):</strong> Apple · Bell Pepper · Blueberry · Cherry · Corn (Maize) · Grape · Orange · Peach · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato
    </div>
    """, unsafe_allow_html=True)
else:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Running MobileNetV2 inference and computing Grad-CAM saliency..."):
            raw_label, clean_label, conf, top3_list, orig_img, heatmap_img, overlay_img = generate_gradcam(img, model, preprocess)

        is_healthy = "healthy" in raw_label.lower()
        pill_html = (
            '<span class="status-pill-healthy">No Pathogen Detected</span>'
            if is_healthy else
            '<span class="status-pill-alert">Pathology Detected — Intervention Recommended</span>'
        )

        # Primary Diagnosis Header Card
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st.markdown(f"""
            <div class="clean-card">
                <div class="card-kicker">Primary Classification</div>
                <div class="diagnosis-heading">{clean_label}</div>
                {pill_html}
            </div>
            """, unsafe_allow_html=True)
        with c_right:
            st.markdown(f"""
            <div class="clean-card" style="text-align: right;">
                <div class="card-kicker">Confidence</div>
                <div class="diagnosis-heading">{conf:.1f}%</div>
                <div style="font-size: 0.78rem; color: #71717A; margin-top: 0.6rem;">MobileNetV2 · 224px</div>
            </div>
            """, unsafe_allow_html=True)

        # Tri-Panel Saliency Grid
        st.markdown('<div class="card-kicker" style="margin-top: 0.5rem;">Visual Attribution (Grad-CAM)</div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(orig_img, caption="Input Specimen (224 × 224)", use_container_width=True)
        with g2:
            st.image(heatmap_img, caption="Activation Saliency Map", use_container_width=True)
        with g3:
            st.image(overlay_img, caption="Lesion Localization Overlay", use_container_width=True)

        # Agronomic Protocol
        pathology_type, organic_rx, chem_rx = get_kvk_advisory(raw_label)
        st.markdown(f'<div class="card-kicker" style="margin-top: 1.5rem;">Agronomic Protocol · {pathology_type}</div>', unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"""
            <div class="protocol-box">
                <div class="protocol-title">Tier 1 — Bio-Control & Organic Management</div>
                <p class="protocol-body">{organic_rx}</p>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="protocol-box">
                <div class="protocol-title">Tier 2 — Targeted Chemical Intervention (KVK)</div>
                <p class="protocol-body">{chem_rx}</p>
            </div>
            """, unsafe_allow_html=True)

        # Subtle Top-3 Breakdown Expander
        with st.expander("View Top-3 Probability Distribution"):
            for cand_label, cand_prob in top3_list:
                st.write(f"**{cand_label}** — {cand_prob:.2f}%")
                st.progress(min(int(cand_prob), 100))

    except Exception as e:
        st.error(f"Could not process image: {e}")
        
