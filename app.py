import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


import cv2
import numpy as np

WORK_SIDE = 640

def limit_side(rgb, max_side=WORK_SIDE):
    """Downscale so the longest side is at most max_side (never upscales)."""
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return rgb

def auto_lesion_mask(rgb, work=256):
    """Rough lesion mask (uint8 0/1, same H x W as rgb): GrabCut seeded with a centred ellipse.
    Falls back to a centred ellipse when GrabCut gives an implausible region."""
    H, W = rgb.shape[:2]
    s = work / max(H, W)
    small = cv2.resize(rgb, (max(16, int(W * s)), max(16, int(H * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    centre = (w // 2, h // 2)
    ok = False
    try:
        m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        cv2.ellipse(m, centre, (int(0.32 * w), int(0.32 * h)), 0, 0, 360, int(cv2.GC_PR_FGD), -1)
        cv2.ellipse(m, centre, (int(0.12 * w), int(0.12 * h)), 0, 0, 360, int(cv2.GC_FGD), -1)
        b = max(2, int(0.04 * min(h, w)))
        m[:b, :] = cv2.GC_BGD; m[-b:, :] = cv2.GC_BGD; m[:, :b] = cv2.GC_BGD; m[:, -b:] = cv2.GC_BGD
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg = (lab == k).astype(np.uint8)
            ok = 0.03 <= float(fg.mean()) <= 0.80
    except Exception:
        ok = False
    if not ok:
        fg = np.zeros((h, w), np.uint8)
        cv2.ellipse(fg, centre, (int(0.30 * w), int(0.30 * h)), 0, 0, 360, 1, -1)
    return cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST)

def crop_to_mask(rgb, mask, margin=0.2, size=320):
    """Square crop around the mask's bounding box (+margin), resized to size x size."""
    H, W = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0) * (1 + 2 * margin)
    side = max(side, 0.25 * max(H, W))
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    ya, yb = max(int(round(cy - side / 2)), 0), min(int(round(cy + side / 2)), H)
    xa, xb = max(int(round(cx - side / 2)), 0), min(int(round(cx + side / 2)), W)
    crop = rgb[ya:yb, xa:xb]
    ch, cw = crop.shape[:2]
    if ch != cw:
        d = abs(ch - cw); a, b = d // 2, d - d // 2
        if ch < cw:
            crop = cv2.copyMakeBorder(crop, a, b, 0, 0, cv2.BORDER_REFLECT_101)
        else:
            crop = cv2.copyMakeBorder(crop, 0, 0, a, b, cv2.BORDER_REFLECT_101)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

def prepare_inputs(raw_rgb, base=320, margin=0.2):
    """Exactly what the model sees: (full image, lesion crop, working-size rgb, lesion mask)."""
    rgb = limit_side(raw_rgb)
    full = cv2.resize(rgb, (base, base), interpolation=cv2.INTER_AREA)
    mask = auto_lesion_mask(rgb)
    crop = crop_to_mask(rgb, mask, margin, base)
    return full, crop, rgb, mask


CFG = json.load(open("config.json"))
BASE, MARGIN, INPUTS, CLASSES = CFG["base"], CFG["margin"], CFG["inputs"], CFG["class_names"]

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras", compile=False)

def eight_views(a):
    out = []
    for flip in (False, True):
        b = a[:, ::-1] if flip else a
        for k in range(4):
            out.append(np.rot90(b, k))
    return np.ascontiguousarray(np.stack(out)).astype("float32")

model = load_model()
st.title("🩺 Skin Lesion Classifier")
st.warning("Educational demo only - this is NOT a medical diagnosis. Please consult a dermatologist.")

file = st.file_uploader("Upload a photo of a skin lesion", type=["jpg", "jpeg", "png"])
if file is not None:
    raw = np.array(Image.open(file).convert("RGB"))
    full, crop, rgb, mask = prepare_inputs(raw, BASE, MARGIN)
    c1, c2 = st.columns(2)
    c1.image(full, caption="Input image")
    c2.image(crop, caption="Detected lesion area")
    views = {"full": eight_views(full), "crop": eight_views(crop)}
    xs = [views[n] for n in INPUTS]
    z = model.predict(xs if len(xs) > 1 else xs[0], verbose=0)          # (8 views, members)
    p1 = float((1.0 / (1.0 + np.exp(-z.mean(axis=0)))).mean())          # mean over members of sigmoid(mean logit)
    probs = [1.0 - p1, p1]
    pred = int(np.argmax(probs))
    st.subheader(f"Prediction: {CLASSES[pred]}  ({probs[pred] * 100:.1f}%)")
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
    st.caption(f"Cross-validated accuracy of this model on the 206-image research dataset: "
               f"{CFG['cv_accuracy'] * 100:.0f}% (+/- 6 points uncertainty). Not validated on real patients.")
