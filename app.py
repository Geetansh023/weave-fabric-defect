from pathlib import Path
import io
import json
import numpy as np
import streamlit as st
import torch
from PIL import Image
from fabric.data import make_image
from fabric.model import load_checkpoint, transform, open_image, GradCAM, overlay

torch.set_num_threads(2)
st.set_page_config(page_title="Weave | Fabric inspection", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:#0b1120;color:#e8edf4}
.block-container {max-width:1180px;padding-top:2.4rem;padding-bottom:2rem}
h1 {font-size:3.3rem!important;font-weight:650!important;letter-spacing:-.06em!important;line-height:1.05!important}
h2,h3 {letter-spacing:-.03em!important}
.kicker {font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#7ee7cf;margin-bottom:16px}
.intro {font-size:1.1rem;color:#a7b5ca;max-width:660px;line-height:1.6;margin:12px 0 24px}
.tag {display:inline-block;background:#172a31;color:#83e6d1;padding:6px 11px;border:1px solid #2c514e;border-radius:20px;font-size:.75rem;margin-right:6px}
.caption {font-size:.8rem;color:#8f9eb5;line-height:1.5}
.panel {background:#121d30;border:1px solid #23314a;border-radius:16px;padding:20px;margin:8px 0 16px}
.label {color:#90a2bc;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase}
.result {font-size:2rem;line-height:1.3;color:#edf4ff;font-weight:650;margin-top:8px}
[data-testid="stSidebar"]{background:#10192a}
[data-testid="stImage"] img {border-radius:12px;image-rendering:auto}
.stButton button {border-radius:10px}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="kicker">WEAVE / VISION LAB</div>', unsafe_allow_html=True)
st.title("A closer look at every thread.")
st.markdown('<div class="intro">Inspect a fabric image, compare the model response, and explore the visual evidence behind it.</div>', unsafe_allow_html=True)
st.markdown('<span class="tag">ResNet18</span><span class="tag">Grad-CAM</span><span class="tag">Local inference</span>', unsafe_allow_html=True)
st.write("")
with st.sidebar:
    st.markdown("## ◈ Weave")
    st.caption("Fabric inspection prototype")
    st.divider()
    source = st.radio("Image source", ["Synthetic samples", "Upload image"])
    if source == "Synthetic samples":
        sample = st.selectbox("Choose sample", ["Synthetic flaw / slash", "Synthetic flaw / spot", "Synthetic intact weave"])
        image = make_image(43 if "slash" in sample else 44, "intact" not in sample)
        image_name = sample
    else:
        upload = st.file_uploader("PNG, JPG or WEBP", type=["png", "jpg", "jpeg", "webp"])
        image = None
        image_name = upload.name if upload else "No image"
        if upload:
            try:
                image = open_image(upload)
            except Exception as exc:
                st.error(f"Cannot read image: {exc}")
    st.divider()
    checkpoint = st.text_input("Local checkpoint", "artifacts/demo.pt")
    opacity = st.slider("Heatmap strength", .1, .9, .55, .05)
    target_choice = st.selectbox("Explain class", ["Model prediction", "defect", "normal"])
    st.caption("Only load checkpoints you trust. No image is sent to an external prediction service by this app.")

model = meta = None
error = None
if Path(checkpoint).is_file():
    try:
        model, meta = load_checkpoint(checkpoint)
    except Exception as exc:
        error = str(exc)
if meta:
    if meta["data_kind"] == "synthetic":
        st.warning("SYNTHETIC DEMO ONLY · This model was trained on generated patterns, not real fabric. Scores do not establish real-world performance.")
    else:
        st.info("User-supplied model. Dataset quality, provenance and generalization have not been independently verified.")
else:
    st.info("No trained checkpoint loaded. View the sample below, then run the quick-start training commands in README.md to enable model inference.")
    if error:
        st.error(error)

if image is not None:
    probabilities = heat = selected = None
    if model:
        cam = GradCAM(model)
        try:
            target = None if target_choice == "Model prediction" else meta["classes"].index(target_choice)
            probabilities, heat, selected = cam(transform(meta["image_size"])(image).unsqueeze(0), target)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
        finally:
            cam.close()
    left, right, summary = st.columns([1, 1, .85], gap="large")
    with left:
        st.markdown("### 01 / Input")
        st.image(image, width="stretch")
        st.caption(f"{image_name} · {image.width} × {image.height} px")
    with right:
        st.markdown("### 02 / Visual evidence")
        if heat is not None:
            result_image = overlay(image, heat, opacity)
            st.image(result_image, width="stretch")
            st.caption(f"Grad-CAM target: {meta['classes'][selected]}. Warm regions influence this class; they are not a segmentation mask.")
            buffer = io.BytesIO()
            result_image.save(buffer, format="PNG")
            st.download_button("Download overlay", buffer.getvalue(), "weave-gradcam.png", "image/png")
        else:
            st.image(image, width="stretch")
            st.caption("Awaiting a trained checkpoint. No heatmap or prediction has been fabricated.")
    with summary:
        st.markdown("### 03 / Model response")
        if probabilities is not None:
            pred = int(probabilities.argmax())
            label = meta["classes"][pred]
            st.markdown(f'<div class="panel"><div class="label">Predicted class</div><div class="result">{label.title()}</div><div class="caption">Synthetic model response</div></div>' if meta['data_kind'] == 'synthetic' else f'<div class="panel"><div class="label">Predicted class</div><div class="result">{label.title()}</div></div>', unsafe_allow_html=True)
            for name, score in zip(meta["classes"], probabilities.tolist()):
                st.caption(f"{name.title()} · {score:.1%}")
                st.progress(float(score))
            st.caption("Softmax scores, not calibrated probabilities or accuracy.")
            record = {"predicted_class": label, "softmax_scores": dict(zip(meta["classes"], probabilities.tolist())), "data_kind": meta["data_kind"], "gradcam_target": meta["classes"][selected]}
            st.download_button("Download prediction JSON", json.dumps(record, indent=2), "prediction.json", "application/json")
        else:
            st.markdown('<div class="panel"><div class="label">Model status</div><div class="result">Not loaded</div><div class="caption">Run training to enable inference.</div></div>', unsafe_allow_html=True)
    st.divider()
    a, b = st.columns([1.4, 1])
    with a:
        st.markdown("#### Read the evidence, not just the label.")
        st.caption("Grad-CAM shows class-sensitive regions at low spatial resolution. A highlighted region can reflect texture, lighting or background, not necessarily a physical flaw.")
    with b:
        st.markdown("#### Built for review")
        st.caption("Fresh implementation · Synthetic starter data · No historical results · No production quality-control claims")
else:
    st.info("Choose a sample or upload an image to begin.")
