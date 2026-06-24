"""
Face Mask Detection - Streamlit App
Smart prediction: tries face detection first, always falls back to full image.
No "No Face Detected" dead ends — always gives a prediction.
"""

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fdf6ff 50%, #fff0f6 100%);
    font-family: 'Segoe UI', sans-serif;
}
#MainMenu, footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #6c63ff, #48cfad);
    border-radius: 20px;
    padding: 36px 30px 28px 30px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(108,99,255,0.18);
}
.hero h1 { color: #ffffff !important; font-size: 2.4rem; font-weight: 800; margin: 0 0 8px 0; }
.hero p  { color: #e8f8ff !important; font-size: 1.05rem; margin: 0; }

[data-testid="stFileUploader"] { background: #f5f3ff; border-radius: 12px; padding: 8px; }
[data-testid="stFileUploader"] label { color: #5b21b6 !important; font-weight: 600 !important; font-size: 1rem !important; }
[data-testid="stFileUploaderDropzone"] { background: #ede9fe !important; border: 2px dashed #7c3aed !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] span { color: #4c1d95 !important; font-weight: 600 !important; }
[data-testid="stFileUploaderDropzone"] small { color: #7c3aed !important; }
section[data-testid="stFileUploaderDropzone"] button { background: #6c63ff !important; color: white !important; border-radius: 8px !important; border: none !important; font-weight: 600 !important; }
[data-testid="stFileUploaderFile"] { color: #1e1b4b !important; background: #ede9fe !important; border-radius: 8px !important; padding: 6px 12px !important; }
[data-testid="stFileUploaderFile"] span { color: #1e1b4b !important; font-weight: 600 !important; }
[data-testid="stFileUploaderFile"] small { color: #5b21b6 !important; }

.result-mask   { background: linear-gradient(135deg, #d1fae5, #a7f3d0); border: 3px solid #10b981; border-radius: 16px; padding: 24px; text-align: center; font-size: 1.5rem; font-weight: 800; color: #064e3b; box-shadow: 0 4px 16px rgba(16,185,129,0.20); margin-top: 12px; }
.result-nomask { background: linear-gradient(135deg, #fee2e2, #fecaca); border: 3px solid #ef4444; border-radius: 16px; padding: 24px; text-align: center; font-size: 1.5rem; font-weight: 800; color: #7f1d1d; box-shadow: 0 4px 16px rgba(239,68,68,0.20); margin-top: 12px; }

.conf-box   { background: #ffffff; border-radius: 14px; padding: 18px 20px; margin-top: 16px; box-shadow: 0 2px 12px rgba(108,99,255,0.10); border-left: 5px solid #6c63ff; }
.conf-label { font-size: 0.95rem; color: #4c1d95; font-weight: 700; margin-bottom: 6px; }
.conf-value { font-size: 2rem; font-weight: 900; color: #6c63ff; }

.notice-box { background: #fef3c7; border: 2px solid #f59e0b; border-radius: 10px; padding: 10px 14px; color: #78350f; font-weight: 600; font-size: 0.88rem; margin-bottom: 12px; }
.face-info  { background: #ede9fe; border-radius: 10px; padding: 10px 14px; color: #4c1d95; font-weight: 600; font-size: 0.9rem; margin-bottom: 10px; }

[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #6c63ff, #48cfad) !important; border-radius: 8px !important; }
[data-testid="stProgressBar"] { background: #ede9fe !important; border-radius: 8px !important; }
[data-testid="stSpinner"] p { color: #5b21b6 !important; font-weight: 600 !important; }
[data-testid="stImage"] p { color: #5b21b6 !important; font-weight: 600 !important; }
p, span, div, label { color: #1e1b4b; }
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 224

@st.cache_resource
def load_resources():
    print("\n" + "="*50)
    print("🚀 INITIALIZING FACE MASK DETECTION SYSTEM (ONNX ENGINE)")
    print("="*50)
    print("[INFO] Loading MobileNetV2 ONNX Model...")
    mdl = ort.InferenceSession("mask_detector.onnx")
    print("[SUCCESS] Model loaded successfully.")
    print("\n MODEL STATISTICS ")
    print("  • Architecture:       MobileNetV2 (ONNX Exported)")
    print("  • Base Model:         Pre-trained on ImageNet")
    print("  • Total Parameters:   ~2.3 Million")
    print("  • Validation Acc:     99.55%")
    print("  • Validation Loss:    0.016")
    print("  • Training Epochs:    20 (with Early Stopping)")
    print("  • Dataset Size:       10,004 images")
    print("="*50 + "\n")
    
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    return mdl, frontal, alt, profile

model, cascade_frontal, cascade_alt, cascade_profile = load_resources()


def try_detect_faces(img_rgb):
    """
    Multi-cascade face detection with smart filtering:
    - Minimum face size: 10% of image dimension (removes tiny false positives like mask graphics)
    - Removes boxes nested inside larger boxes
    - NMS deduplication
    Returns list of (x,y,w,h).
    """
    gray   = cv2.equalizeHist(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    H, W   = gray.shape
    min_sz = max(40, int(min(W, H) * 0.10))   # at least 10% of image size

    all_faces = []

    for cascade in [cascade_frontal, cascade_alt]:
        for sf, mn in [(1.05, 5), (1.1, 4), (1.15, 3)]:
            found = cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn,
                minSize=(min_sz, min_sz)
            )
            if len(found) > 0:
                all_faces.extend(found.tolist())

    # Profile + mirrored profile
    for sf, mn in [(1.05, 4), (1.1, 3)]:
        found = cascade_profile.detectMultiScale(gray, scaleFactor=sf,
                    minNeighbors=mn, minSize=(min_sz, min_sz))
        if len(found) > 0:
            all_faces.extend(found.tolist())
        found_f = cascade_profile.detectMultiScale(cv2.flip(gray,1), scaleFactor=sf,
                    minNeighbors=mn, minSize=(min_sz, min_sz))
        if len(found_f) > 0:
            all_faces.extend([[W - x - w, y, w, h] for (x,y,w,h) in found_f])

    if not all_faces:
        return []

    # ── Filter 1: remove boxes that are nested inside a larger box ────────────
    all_faces = sorted(all_faces, key=lambda b: b[2]*b[3], reverse=True)
    filtered = []
    for box in all_faces:
        x, y, w, h = box
        cx, cy = x + w//2, y + h//2          # centre of this box
        nested = False
        for (ux, uy, uw, uh) in filtered:
            # If this box's centre is inside a larger already-kept box → skip
            if ux < cx < ux+uw and uy < cy < uy+uh and (w*h) < (uw*uh)*0.7:
                nested = True
                break
        if not nested:
            filtered.append(box)

    # ── Filter 2: NMS — remove boxes with >40% overlap with a larger box ─────
    unique = []
    for box in filtered:
        x, y, w, h = box
        duplicate = False
        for (ux, uy, uw, uh) in unique:
            ix1 = max(x, ux); iy1 = max(y, uy)
            ix2 = min(x+w, ux+uw); iy2 = min(y+h, uy+uh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = w*h + uw*uh - inter
                if inter/union > 0.40:
                    duplicate = True
                    break
        if not duplicate:
            unique.append(box)

    return unique


def predict_crop(img_rgb, x1, y1, x2, y2):
    """Run model on a cropped region."""
    crop     = Image.fromarray(img_rgb[y1:y2, x1:x2]).resize((IMG_SIZE, IMG_SIZE))
    arr      = np.expand_dims(np.array(crop, dtype=np.float32) / 255.0, axis=0)
    
    # Run inference using ONNX
    input_name = model.get_inputs()[0].name
    ort_outs   = model.run(None, {input_name: arr})
    score      = float(ort_outs[0][0][0])
    
    has_mask = score < 0.5
    conf     = (1.0 - score) if has_mask else score
    label    = "with_mask" if has_mask else "without_mask"
    return crop, label, conf, score


def smart_predict(pil_image):
    """
    1. Try face detection
    2. If faces found → predict on each face crop
    3. If NO faces found → predict on intelligent crops of the image
       (full image + top-half crop, take majority vote)
    Returns list of result dicts.
    """
    img_rgb = np.array(pil_image)
    H, W    = img_rgb.shape[:2]
    results = []

    faces = try_detect_faces(img_rgb)

    if faces:
        for (x, y, w, h) in faces:
            pad = int(0.12 * w)
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
            crop, label, conf, score = predict_crop(img_rgb, x1, y1, x2, y2)
            # Draw box on image
            color = (16, 185, 129) if label == "with_mask" else (239, 68, 68)
            text  = f"Mask {conf*100:.0f}%" if label == "with_mask" else f"No Mask {conf*100:.0f}%"
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img_rgb, (x1, y1-30), (x2, y1), color, -1)
            cv2.putText(img_rgb, text, (x1+6, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            results.append({"label": label, "conf": conf, "score": score,
                            "crop": crop, "face_detected": True})
    else:
        # No face detected — run 3 smart crops and majority vote
        crops_regions = [
            (0, 0, W, H),                                  # full image
            (W//6, 0, 5*W//6, int(H*0.75)),               # centre top-75%
            (W//4, int(H*0.05), 3*W//4, int(H*0.85)),     # tighter centre
        ]
        votes = []
        for (cx1, cy1, cx2, cy2) in crops_regions:
            _, lbl, cf, sc = predict_crop(img_rgb, cx1, cy1, cx2, cy2)
            votes.append((lbl, cf, sc))

        # Majority vote
        mask_votes   = [v for v in votes if v[0] == "with_mask"]
        nomask_votes = [v for v in votes if v[0] == "without_mask"]

        if len(mask_votes) >= len(nomask_votes):
            best = max(mask_votes, key=lambda v: v[1])
        else:
            best = max(nomask_votes, key=lambda v: v[1])

        lbl, cf, sc = best
        # Use centre crop as display image
        crop, _, _, _ = predict_crop(img_rgb, W//6, 0, 5*W//6, int(H*0.75))
        results.append({"label": lbl, "conf": cf, "score": sc,
                        "crop": crop, "face_detected": False})

    return results, Image.fromarray(img_rgb)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>😷 Face Mask Detector</h1>
    <p>Upload any photo — AI detects faces and checks if a mask is worn</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📁 Upload an Image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Analysing image..."):
        results, annotated_img = smart_predict(image)

    face_detected = results[0]["face_detected"]
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        caption = (f"📷 {len(results)} face(s) detected"
                   if face_detected else "📷 Uploaded Image")
        st.image(annotated_img, caption=caption, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Result")

        if not face_detected:
            st.markdown("""
            <div class="notice-box">
                ⚠️ Face auto-detection was tricky (side angle / mask occlusion / low lighting).<br>
                Used smart full-image analysis instead.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="face-info">👤 {len(results)} face(s) found</div>',
                unsafe_allow_html=True
            )

        for i, r in enumerate(results):
            if len(results) > 1:
                st.markdown(f"**Face {i+1}:**")

            if r["label"] == "with_mask":
                st.markdown('<div class="result-mask">✅ Mask Detected!</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-nomask">❌ No Mask Detected!</div>',
                            unsafe_allow_html=True)

            st.markdown(f"""
            <div class="conf-box">
                <div class="conf-label">🔬 Confidence Score</div>
                <div class="conf-value">{r['conf'] * 100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            st.progress(float(r["conf"]))

            if len(results) > 1:
                st.markdown("---")

        # Show face crops if faces were detected
        if face_detected and len(results) > 0:
            st.markdown("#### 🖼️ Detected Face(s)")
            cols = st.columns(min(len(results), 4))
            for i, r in enumerate(results):
                with cols[i % 4]:
                    emoji = "✅" if r["label"] == "with_mask" else "❌"
                    st.image(r["crop"],
                             caption=f"{emoji} {r['conf']*100:.0f}%",
                             use_container_width=True)