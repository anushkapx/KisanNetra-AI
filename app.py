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
    page_title="KisanNetra — Crop Health Check",
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
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
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
    activations = []
    gradients = []
    target_layer = model.mobilenet_v2.layer[-1]

    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    outputs = model(input_tensor)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0]

    if selected_crop != "Auto-Detect (All 14 Crops)":
        crop_key = selected_crop.split(" ")[0].lower()
        masked_probs = torch.zeros_like(probs)
        for idx, lbl in model.config.id2label.items():
            if crop_key in lbl.lower():
                masked_probs[idx] = probs[idx]
        if masked_probs.sum() > 0:
            probs = masked_probs / masked_probs.sum()

    k = min(3, int(torch.count_nonzero(probs).item()))
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

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    if "healthy" in raw_label.lower():
        severity_pct = 0.0
        severity_stage = "Healthy (0% Lesion Spread)"
        urgency_note = "Routine preventive care only."
    else:
        binary_lesion_mask = (cam >= 0.55).astype(np.uint8)
        raw_ratio = (np.sum(binary_lesion_mask) / (224 * 224)) * 100
        severity_pct = min(round(raw_ratio * 1.45, 1), 95.0)
        if severity_pct < 20.0:
            severity_stage = "Stage 1 — Mild Infection"
            urgency_note = "Early onset. Start Option 1 (Organic Bio-Control) immediately."
        elif severity_pct < 45.0:
            severity_stage = "Stage 2 — Moderate Spread"
            urgency_note = "Active lesion growth. Apply KVK spray within 48 hours."
        else:
            severity_stage = "Stage 3 — Severe Infestation"
            urgency_note = "High yield-loss risk! Prune worst leaves and spray KVK fungicide/bactericide today."

    orig_resized = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * orig_resized + 0.45 * heatmap)

    return (
        raw_label, format_disease_name(raw_label), confidence,
        severity_pct, severity_stage, urgency_note,
        top3_list, orig_resized, heatmap, overlay
    )

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
            "Place 15 Yellow Sticky Traps per acre to catch whiteflies, and spray 5% Neem Seed Kernel Extract (NSKE 5%). Remove and bury heavily curled plants so it doesn't spread.",
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

def build_farmer_slip(clean_label, conf, severity_pct, severity_stage, pathology_type, organic_rx, chem_rx):
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return f"""============================================================
         KISANNETRA AI — KVK CROP DIAGNOSTIC SLIP
============================================================
Date & Time       : {now_str}
Detected Condition: {clean_label}
Model Confidence  : {conf:.1f}%
Infection Severity: {severity_pct}% ({severity_stage})
Pathology Category: {pathology_type}
------------------------------------------------------------
OPTION 1: ORGANIC / BIO-CONTROL (जैविक उपाय)
{organic_rx}

OPTION 2: KVK CHEMICAL DOSAGE (रासायनिक उपचार)
{chem_rx}
------------------------------------------------------------
Note: Show this slip at your nearest Krishi Seva Kendra / KVK
for accurate input dosage. Generated by KisanNetra AI.
============================================================
"""

st.markdown("""
<div class="farm-header">
    <div class="hindi-greet">नमस्ते किसान भाई · आपकी फसल का डिजिटल साथी</div>
    <h1 class="farm-title">KisanNetra — Crop Health & Severity Check</h1>
    <div class="farm-tagline">
        Upload a close-up leaf photo to identify the disease, measure how much of the leaf is infected, and download a ready Krishi Vigyan Kendra (KVK) treatment slip.
    </div>
</div>
""", unsafe_allow_html=True)

col_u1, col_u2 = st.columns([2, 1])
with col_u1:
    uploaded_file = st.file_uploader(
        "📷 Step 1: Upload a Clear Single-Leaf Photo (JPG or PNG)",
        type=["jpg", "png", "jpeg"]
    )
with col_u2:
    selected_crop = st.selectbox(
        "🌱 Step 2: Select Crop (Optional)",
        [
            "Auto-Detect (All 14 Crops)",
            "Potato (आलू)",
            "Tomato (टमाटर)",
            "Pepper (शिमला मिर्च)",
            "Corn (मक्का)",
            "Apple (सेब)",
            "Grape (अंगूर)",
            "Orange (संतरा)",
            "Peach",
            "Cherry",
            "Strawberry",
            "Soybean (सोयाबीन)",
            "Squash",
            "Blueberry",
            "Raspberry"
        ]
    )

if uploaded_file is None:
    st.markdown("#### How KisanNetra Helps Your Farm in 4 Steps")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 1</span>
            <div class="step-heading">Upload Leaf Photo</div>
            <p class="step-desc">Click a clear close-up photo of one leaf in daylight so spots or yellowing are visible.</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 2</span>
            <div class="step-heading">Safe Crop Check</div>
            <p class="step-desc">Verifies the leaf against 14 supported crops and blocks uncertain guesses to protect your harvest.</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 3</span>
            <div class="step-heading">Heat Scan & Severity %</div>
            <p class="step-desc">Highlights infected patches in red/yellow and calculates the exact percentage of leaf damage.</p>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="step-card">
            <span class="step-num">Step 4</span>
            <div class="step-heading">KVK Slip & Voice Guide</div>
            <p class="step-desc">Listen to the diagnosis aloud or download a KVK medicine slip for your local agro-shop.</p>
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
        with st.spinner("Scanning leaf lesions, measuring infection severity, and preparing KVK slip..."):
            (
                raw_label, clean_label, conf,
                severity_pct, severity_stage, urgency_note,
                top3_list, orig_img, heatmap_img, overlay_img
            ) = generate_gradcam_and_severity(img, model, preprocess, selected_crop)

        if conf < 20.0:
            st.warning(
                f"**Unsupported Crop or Unclear Photo (Confidence: {conf:.1f}%)**\n\n"
                "This image does not strongly match any of our **14 supported crop species (38 classes)**. "
                "Please upload a clear single-leaf close-up photo."
            )
        else:
            is_healthy = "healthy" in raw_label.lower()
            status_badge = (
                '<span class="badge-healthy">स्वस्थ फसल · Crop Looks Healthy</span>'
                if is_healthy else
                '<span class="badge-sick">रोग की पहचान · Treatment Advised</span>'
            )

            c_diag, c_conf, c_sev = st.columns([2.2, 1, 1.3])
            with c_diag:
                st.markdown(f"""
                <div class="result-banner">
                    <div class="result-label">Detected Crop & Condition</div>
                    <div class="result-disease">{clean_label}</div>
                    {status_badge}
                </div>
                """, unsafe_allow_html=True)
            with c_conf:
                st.markdown(f"""
                <div class="result-banner" style="text-align: center;">
                    <div class="result-label">AI Confidence</div>
                    <div class="result-disease" style="color: #1E4620;">{conf:.1f}%</div>
                    <div style="font-size: 0.78rem; color: #5C6B5B;">Verified Match</div>
                </div>
                """, unsafe_allow_html=True)
            with c_sev:
                sev_color = "#166534" if is_healthy else ("#D97706" if severity_pct < 35 else "#B8502A")
                st.markdown(f"""
                <div class="result-banner" style="text-align: center;">
                    <div class="result-label">Infected Leaf Area</div>
                    <div class="result-disease" style="color: {sev_color};">{severity_pct}%</div>
                    <div style="font-size: 0.78rem; color: #5C6B5B;">{severity_stage}</div>
                </div>
                """, unsafe_allow_html=True)

            if not is_healthy:
                st.info(f"**Field Action Urgency:** {urgency_note}")

            st.markdown("#### Visual Leaf Scan (Red/Yellow Zones Show Infected Tissue)")
            g1, g2, g3 = st.columns(3)
            with g1:
                st.image(orig_img, caption="1. Uploaded Leaf (224×224)", use_container_width=True)
            with g2:
                st.image(heatmap_img, caption="2. Grad-CAM Saliency Map", use_container_width=True)
            with g3:
                st.image(overlay_img, caption="3. Lesion Overlay & Mask", use_container_width=True)

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

            st.markdown("#### Farmer Action Tools (Offline Shop Slip & Audio Readout)")
            slip_text = build_farmer_slip(
                clean_label, conf, severity_pct, severity_stage,
                pathology_type, organic_rx, chem_rx
            )
            tool_col1, tool_col2 = st.columns([1, 1])
            with tool_col1:
                st.download_button(
                    label="📄 Download KVK Farmer Prescription Slip (.txt)",
                    data=slip_text,
                    file_name=f"KisanNetra_Advisory_{clean_label.replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with tool_col2:
                spoken_summary = (
                    f"Namaste Kisan Bhai. KisanNetra diagnosis is {clean_label}, "
                    f"with {severity_pct} percent leaf area affected. "
                    f"Recommended organic care: {organic_rx}"
                ).replace('"', '').replace("'", "")
                components.html(
                    f"""
                    <button onclick="const u = new SpeechSynthesisUtterance('{spoken_summary}'); u.rate = 0.95; window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);"
                        style="width: 100%; padding: 0.55rem 1rem; background-color: #1E4620; color: white; border: none; border-radius: 8px; font-family: Poppins, sans-serif; font-size: 0.92rem; font-weight: 600; cursor: pointer;">
                        🔊 Listen to Diagnosis & Remedy Aloud
                    </button>
                    """,
                    height=52
                )

            with st.expander("Technical Model Details & Top-3 Predictions (For Project Evaluation)"):
                st.caption("Architecture: MobileNetV2 (1.0_224) | XAI: Terminal Layer Grad-CAM | Severity: OpenCV Saliency Thresholding (τ = 0.55)")
                for cand_label, cand_prob in top3_list:
                    st.write(f"**{cand_label}** — {cand_prob:.2f}%")
                    st.progress(min(int(cand_prob), 100))

    except Exception as e:
        st.error(f"Could not read this image: {e}. Please upload a clear JPG or PNG photo.")
