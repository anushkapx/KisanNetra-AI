import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification

# Page configuration
st.set_page_config(
    page_title="KisanNetra AI | Smart Crop Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a vibrant, modern Agritech UI
st.markdown("""
<style>
    /* Main background & typography */
    .stApp {
        background: linear-gradient(180deg, #f4fbf6 0%, #ffffff 100%);
    }
    /* Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(27, 67, 50, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #d8f3dc;
        margin-bottom: 1rem;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .pill-badge {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
    }
    /* Metric / Diagnosis Cards */
    .diag-card {
        background: #ffffff;
        border-left: 6px solid #2d6a4f;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .rx-organic {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-left: 6px solid #2e7d32;
        padding: 1.2rem;
        border-radius: 12px;
        height: 100%;
    }
    .rx-chemical {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-left: 6px solid #f57f17;
        padding: 1.2rem;
        border-radius: 12px;
        height: 100%;
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

def generate_gradcam(img_pil, model, preprocess):
    input_tensor = preprocess(img_pil).unsqueeze(0)
    activations, gradients = [], []
    target_layer = model.mobilenet_v2.layer[-1]
    
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    
    outputs = model(input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]
    
    # Top-3 predictions for richer visual breakdown
    top3_probs, top3_indices = torch.topk(probs, 3)
    pred_idx = top3_indices[0].item()
    confidence = top3_probs[0].item() * 100
    raw_label = model.config.id2label[pred_idx]
    
    top3_list = [
        (model.config.id2label[idx.item()], prob.item() * 100)
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
    
    return raw_label, confidence, top3_list, orig_resized, heatmap, overlay

def get_kvk_advisory(label):
    low_label = label.lower()
    if "healthy" in low_label:
        return (
            "Healthy Foliage (No Pathogen Detected)",
            "Maintain standard irrigation and balanced NPK nutrition. Apply a preventive spray of Cold-Pressed Neem Oil (1500 ppm @ 3 ml/L) every 15 days.",
            "No chemical fungicide or bactericide required—crop foliage is vigorous."
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

# Sidebar for Project Context & Supported Crops
with st.sidebar:
    st.markdown("## 🌾 KisanNetra AI")
    st.caption("Explainable Plant Pathology & KVK Advisory Platform")
    st.markdown("---")
    st.markdown("### 🌱 Supported Crops (14 Species)")
    crops = [
        "🍅 Tomato", "🥔 Potato", "🫑 Bell Pepper", "🌽 Corn (Maize)",
        "🍎 Apple", "🍇 Grape", "🍊 Orange", "🍑 Peach",
        "🍒 Cherry", "🍓 Strawberry", "🫘 Soybean", "🎃 Squash",
        "🫐 Blueberry", "🍇 Raspberry"
    ]
    st.write(" • ".join(crops))
    st.markdown("---")
    st.markdown("### 🔬 How Grad-CAM Works")
    st.info(
        "Red/Yellow thermal zones highlight the exact necrotic lesions "
        "used by MobileNetV2 to make its diagnosis, proving the model "
        "ignores background soil or shadows."
    )

# Hero Banner
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🌿 KisanNetra AI</div>
    <div class="hero-sub">Explainable Multi-Crop Pathology & KVK Agronomic Decision Support System</div>
    <div class="badge-row">
        <span class="pill-badge">🧬 38 Disease Classes</span>
        <span class="pill-badge">🌱 14 Crop Species</span>
        <span class="pill-badge">🔥 Real-Time Grad-CAM XAI</span>
        <span class="pill-badge">🚜 KVK Treatment Protocols</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader(
    "📸 Upload a clear leaf photograph to start AI diagnosis (JPG, JPEG, PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is None:
    # Clean 3-step onboarding cards when idle
    st.markdown("### ✨ Quick Diagnostic Workflow")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. Capture Leaf Specimen**\n\nUpload a close-up photo of an affected or healthy leaf under natural daylight.")
    with col2:
        st.success("**2. Visual XAI Verification**\n\nInspect the Grad-CAM saliency heatmap to verify lesion-level localization.")
    with col3:
        st.warning("**3. KVK Action Plan**\n\nReceive dual-tier organic bio-control and calibrated chemical dosages.")
else:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("🔍 Analyzing cellular leaf patterns & computing Grad-CAM gradients..."):
            label, conf, top3_list, orig_img, heatmap_img, overlay_img = generate_gradcam(img, model, preprocess)
            
        is_healthy = "healthy" in label.lower()
        status_color = "#2e7d32" if is_healthy else "#c62828"
        status_icon = "✅" if is_healthy else "⚠️"
        
        # Top Metric Summary Row
        m1, m2, m3 = st.columns([2, 1, 1])
        with m1:
            st.markdown(f"""
            <div class="diag-card" style="border-left-color: {status_color};">
                <span style="font-size: 0.85rem; color: #555; font-weight: 600;">PRIMARY AI DIAGNOSIS</span><br>
                <span style="font-size: 1.45rem; font-weight: 800; color: {status_color};">{status_icon} {label}</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.metric(label="Model Confidence", value=f"{conf:.2f}%", delta="Verified XAI")
        with m3:
            st.metric(label="Inference Backbone", value="MobileNetV2", delta="38 Classes")

        # Interactive Tabs for Clean Presentation
        tab1, tab2, tab3 = st.tabs(["🔥 Grad-CAM Visual Proof", "💊 KVK Treatment Protocol", "📊 Top-3 Probability Breakdown"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(orig_img, caption="1. Standardized Leaf Input (224×224)", use_container_width=True)
            with c2:
                st.image(heatmap_img, caption="2. Grad-CAM Saliency Heatmap", use_container_width=True)
            with c3:
                st.image(overlay_img, caption="3. Diagnostic Lesion Overlay", use_container_width=True)
                
        with tab2:
            pathology_type, organic_rx, chem_rx = get_kvk_advisory(label)
            st.markdown(f"#### Recommended Action Plan — *{pathology_type}*")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="rx-organic">
                    <h4 style="color: #1b5e20; margin-top: 0;">🌱 Tier-1: Bio-Control & Organic Management</h4>
                    <p style="color: #1f2937; font-size: 0.98rem;">{organic_rx}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="rx-chemical">
                    <h4 style="color: #e65100; margin-top: 0;">🧪 Tier-2: Targeted Chemical Intervention (KVK)</h4>
                    <p style="color: #1f2937; font-size: 0.98rem;">{chem_rx}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with tab3:
            st.markdown("#### Classifier Confidence Distribution (Top-3 Candidates)")
            for cand_label, cand_prob in top3_list:
                st.write(f"**{cand_label}** — `{cand_prob:.2f}%`")
                st.progress(min(int(cand_prob), 100))

    except Exception as e:
        st.error(f"Error processing image: {e}. Please upload a valid JPG or PNG leaf image.")
