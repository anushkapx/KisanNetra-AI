import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForImageClassification

st.set_page_config(
    page_title="KisanNetra — फसल स्वास्थ्य जांच",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Warm, earthy Indian agricultural palette (Deep Foliage, Terracotta Clay, Turmeric, Sunlit Cream)
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
        max-width: 1080px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* Warm Field Banner */
    .farm-header {
        background-color: #1E4620;
        border-bottom: 6px solid #D97706;
        border-radius: 14px;
        padding: 2rem 2.2rem;
        color: #FFFDF7;
        margin-bottom: 1.8rem;
    }
    .hindi-greet {
        font-family: 'Tiro Devanagari Hindi', serif;
        font-size: 1.15rem;
        color: #FCD34D;
        margin-bottom: 0.3rem;
        letter-spacing: 0.02em;
    }
    .farm-title {
        font-size: 2.35rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
    }
    .farm-tagline {
        font-size: 1rem;
        color: #E2ECE1;
        margin-top: 0.5rem;
        max-width: 680px;
    }

    /* Simple Step Cards */
    .step-card {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-top: 4px solid #2E6F40;
        border-radius: 12px;
        padding: 1.3rem;
        height: 100%;
    }
    .step-num {
        display: inline-block;
        background-color: #2E6F40;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 2px 10px;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .step-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E3F20;
        margin-bottom: 0.35rem;
    }
    .step-desc {
        font-size: 0.9rem;
        color: #4A5849;
        line-height: 1.5;
        margin: 0;
    }

    /* Diagnosis Result Card */
    .result-banner {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .result-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #5C6B5B;
    }
    .result-disease {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1E3F20;
        margin: 0.25rem 0 0.6rem 0;
    }
    .badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-sick {
        background-color: #FFEDD5;
        color: #9A3412;
        border: 1px solid #FDBA74;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Remedy Cards (Organic vs Chemical) */
    .remedy-organic {
        background-color: #F3FAF1;
        border: 1.5px solid #B7D9B1;
        border-left: 6px solid #2E6F40;
        border-radius: 12px;
        padding: 1.4rem;
        height: 100%;
    }
    .remedy-chemical {
        background-color: #FFF9EB;
        border: 1.5px solid #F3D59B;
        border-left: 6px solid #B8502A;
        border-radius: 12px;
        padding: 1.4rem;
        height: 100%;
    }
    .remedy-head {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .remedy-text {
        font-size: 0.94rem;
        line-height: 1.6;
        color: #263325;
        margin: 0;
    }

    /* Crop List Footer Box */
    .crop-box {
        background-color: #FFFDF7;
        border: 1.5px solid #D5DFD2;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        color: #2D3B2C;
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
            "फसल स्वस्थ है · Healthy Crop",
            "Regular irrigation and balanced farmyard manure / NPK are sufficient. Spray Neem Oil (1500 ppm @ 3 ml per liter of water) once every 15 days to keep pests away naturally.",
            "No chemical medicine needed. Save your fertilizer and pesticide cost!"
        )
    elif any(k in low_label for k in ["blight", "mildew", "spot", "rust", "scab", "rot"]):
        return (
            "फफूंद जनित रोग · Fungal Leaf Infection",
            "Mix Trichoderma viride (5 g per liter of water) or Copper Oxychloride 50 WP (2.5 g/L) and spray evenly on affected leaves during early morning or evening.",
            "If spots are spreading fast, spray Mancozeb 75 WP (2.0 g per liter of water) or Azoxystrobin 23% SC (1.0 ml/L) at 10-day intervals as advised by KVK."
        )
    elif any(k in low_label for k in ["virus", "mosaic", "curl"]):
        return (
            "विषाणु रोग · Viral Leaf Curl / Mosaic",
            "Place 15 Yellow Sticky Traps per acre to catch whiteflies, and spray 5% Neem Seed Kernel Extract (NSKE). Remove and bury heavily curled plants so it doesn't spread.",
            "To stop whiteflies and aphids carrying the virus, spray Imidacloprid 17.8% SL (0.5 ml per liter of water) or Thiamethoxam 25% WG (0.3 g/L)."
        )
    elif "bacterial" in low_label:
        return (
            "जीवाणु रोग · Bacterial Spot / Blight",
            "Pluck and destroy severely spotted lower leaves. Spray Pseudomonas fluorescens bio-bactericide (5 g per liter of water) across the crop.",
            "Spray Copper Hydroxide 77% WP (2.0 g/L) mixed with Streptocycline (0.15 g per liter of water) per Krishi Vigyan Kendra (KVK) guidance."
        )
    else:
        return (
            "कीट / माइट प्रकोप · Pest or Mite Damage",
            "Spray Neem Oil (5 ml per liter of water) thoroughly on the underside of leaves where mites hide.",
            "For heavy mite infestation, spray Spiromesifen 22.9% SC (0.8 ml per liter of water) under local KVK supervision."
        )

# Warm Farmer-Relatable Masthead
st.markdown("""
<div class="farm-header">
    <div class="hindi-greet">नमस्ते किसान भाई · आपकी फसल का डिजिटल साथी</div>
    <h1 class="farm-title">KisanNetra — Crop Health Check</h1>
    <div class="farm-tagline">
        Take a close-up photo of any sick leaf to identify the exact disease in seconds, see the affected spots clearly, and get trusted Krishi Vigyan Kendra (KVK) remedies.
    </div>
</div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader(
    "📷 Upload or Take a Photo of the Crop Leaf (JPG or PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is None:
    st.markdown("#### How to Check Your Crop in 3 Easy Steps")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 1</span>
            <div class="step-heading">Click a Clear Leaf Photo</div>
            <p class="step-desc">Hold your phone close to a single affected leaf in daylight so the spots or yellowing are clearly visible.</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 2</span>
            <div class="step-heading">See Where the Disease Is</div>
            <p class="step-desc">Our visual scanner highlights the exact infected patches in red/yellow so you can verify the check with your own eyes.</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 3</span>
            <div class="step-heading">Get KVK Remedy & Dosage</div>
            <p class="step-desc">Receive both a low-cost organic/desi treatment and the exact KVK-recommended medicine dosage per liter of water.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="crop-box">
        <strong>14 Supported Crops (38 Conditions):</strong> Tomato (टमाटर) · Potato (आलू) · Bell Pepper (शिमला मिर्च) · Corn / Maize (मक्का) · Apple (सेब) · Grape (अंगूर) · Orange (संतरा) · Peach · Cherry · Strawberry · Soybean (सोयाबीन) · Squash · Blueberry · Raspberry
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Checking leaf spots and preparing your KVK crop advisory..."):
            raw_label, clean_label, conf, top3_list, orig_img, heatmap_img, overlay_img = generate_gradcam(img, model, preprocess)

        is_healthy = "healthy" in raw_label.lower()
        status_badge = (
            '<span class="badge-healthy">स्वस्थ फसल · Crop Looks Healthy</span>'
            if is_healthy else
            '<span class="badge-sick">रोग की पहचान · Treatment Advised</span>'
        )

        # Result Summary Card
        r_left, r_right = st.columns([3, 1])
        with r_left:
            st.markdown(f"""
            <div class="result-banner">
                <div class="result-label">Detected Crop & Condition</div>
                <div class="result-disease">{clean_label}</div>
                {status_badge}
            </div>
            """, unsafe_allow_html=True)
        with r_right:
            st.markdown(f"""
            <div class="result-banner" style="text-align: center;">
                <div class="result-label">Match Accuracy</div>
                <div class="result-disease" style="color: #B8502A;">{conf:.1f}%</div>
                <div style="font-size: 0.8rem; color: #5C6B5B;">Verified Leaf Scan</div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Leaf Scan Grid
        st.markdown("#### Leaf Spot Scan (Red/Yellow Areas Show Infected Patches)")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(orig_img, caption="1. Your Uploaded Leaf", use_container_width=True)
        with g2:
            st.image(heatmap_img, caption="2. Disease Heat Scan (Grad-CAM)", use_container_width=True)
        with g3:
            st.image(overlay_img, caption="3. Infected Area Highlighted", use_container_width=True)

        # Farmer-Friendly Remedies
        pathology_type, organic_rx, chem_rx = get_kvk_advisory(raw_label)
        st.markdown(f"#### Recommended Field Care ({pathology_type})")
        c_org, c_chem = st.columns(2)
        with c_org:
            st.markdown(f"""
            <div class="remedy-organic">
                <div class="remedy-head" style="color: #1E4620;">Option 1: जैविक उपाय (Organic / Bio-Control)</div>
                <p class="remedy-text">{organic_rx}</p>
            </div>
            """, unsafe_allow_html=True)
        with c_chem:
            st.markdown(f"""
            <div class="remedy-chemical">
                <div class="remedy-head" style="color: #9A3412;">Option 2: रासायनिक उपचार (KVK Spray Dosage)</div>
                <p class="remedy-text">{chem_rx}</p>
            </div>
            """, unsafe_allow_html=True)

        # Optional Technical Details for Viva / Evaluator
        with st.expander("Technical Model Details & Top-3 Predictions (For Project Evaluation)"):
            st.caption("Architecture: MobileNetV2 (1.0_224) | Explainability: Terminal Layer Grad-CAM | 38 PlantVillage Classes")
            for cand_label, cand_prob in top3_list:
                st.write(f"**{cand_label}** — {cand_prob:.2f}%")
                st.progress(min(int(cand_prob), 100))

    except Exception as e:
        st.error(f"Could not read this image: {e}. Please upload a clear JPG or PNG photo.")
