"""
Cow Weight Farmville — Flask backend v2
Pipeline: YOLOv8n-seg → morphometric features → Ridge regression
"""
import cv2
import os
import math
import tempfile
import traceback

import numpy as np
import pandas as pd
import joblib
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PKL   = os.environ.get("MODEL_PATH",   "model/cow_weight_model.pkl")
SEG_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "model/yolov8n_seg_cattle.pt")

print("Loading models...")
pipeline  = joblib.load(MODEL_PKL)
seg_model = YOLO(SEG_WEIGHTS)
print(f"  Pipeline: {MODEL_PKL}")
print(f"  Segmentation: {SEG_WEIGHTS}")

INFER_SIZE = 640  # resize longest edge to this before feature extraction (match training)
STICKER_REAL_CM = 10.0
CM_TO_INCH      = 0.393701
LBS_TO_KG       = 0.453592

# ── Feature extraction ────────────────────────────────────────────────────────


# ── Sticker → px/cm ───────────────────────────────────────────────────────

def get_px_per_cm(ann_side_path):
    if not ann_side_path or not os.path.exists(str(ann_side_path)):
        return None
    ann_rgb    = cv2.cvtColor(cv2.imread(str(ann_side_path)), cv2.COLOR_BGR2RGB)
    s_mask     = make_colour_mask(ann_rgb, STICKER_RGB)
    sticker_px = int(s_mask.sum())
    if sticker_px > 50:
        return np.sqrt(sticker_px) / STICKER_REAL_CM   # sticker is 10×10cm square
    return None

# ── Body trunk at a column ────────────────────────────────────────────────
# Finds the LARGEST continuous vertical run of cattle pixels.
# The body (wide, continuous blob) separates from the legs (thin, lower blobs)
# by a small gap in the mask column — we keep only the main body segment.

def get_body_trunk_at_col(pred_mask, col):
    col_pixels  = pred_mask[:, col]
    cattle_rows = np.where(col_pixels == 1)[0]
    if len(cattle_rows) < 2:
        return None, None
    runs, start, prev = [], cattle_rows[0], cattle_rows[0]
    for r in cattle_rows[1:]:
        if r - prev > 15:          # gap > 15px = separate segment (leg)
            runs.append((start, prev))
            start = r
        prev = r
    runs.append((start, prev))
    best = max(runs, key=lambda x: x[1] - x[0])
    return int(best[0]), int(best[1])   # trunk_top, trunk_bottom

# ── Body length: diagonal A→B ─────────────────────────────────────────────
# A = point of shoulder  (front of body, lower — ~15% from left of bbox)
# B = pin bone / rump    (rear of body, upper — scan from right, exclude thin tail)

def get_body_length_diagonal(pred_mask, cmin, cmax):
    bbox_w = cmax - cmin

    # Find body region: rows with max horizontal span (the barrel, not head/legs)
    row_widths = np.sum(pred_mask[:, cmin:cmax+1], axis=1)
    max_width = row_widths.max()
    if max_width == 0:
        return 0, (cmin, cmax//2), (cmax, cmax//2)

    # Keep only rows at least 70% of max width (excludes thin head/leg rows)
    body_rows = np.where(row_widths > max_width * 0.70)[0]
    if len(body_rows) < 10:
        return 0, (cmin, cmax//2), (cmax, cmax//2)

    body_rmin, body_rmax = body_rows[0], body_rows[-1]
    body_mask = pred_mask[body_rmin:body_rmax+1, cmin:cmax+1]

    # Now find A and B within body region only
    col_A = cmin + int(bbox_w * 0.15)
    top_A, bot_A = get_body_trunk_at_col(body_mask, col_A - cmin)
    if bot_A is None:
        bot_A = body_rmax - body_rmin
    A = (col_A, body_rmin + bot_A)

    col_B = cmax
    for col in range(cmax, cmin + int(bbox_w * 0.55), -1):
        t, b = get_body_trunk_at_col(body_mask, col - cmin)
        if t is not None and (b - t) > 30:
            col_B, top_B = col, body_rmin + t
            break
    B = (col_B, top_B if 'top_B' in locals() else body_rmin)

    length_px = float(np.sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2))
    return length_px, A, B
# ── Girth M1: side view — body trunk height at shoulder (no legs) ─────────

def get_girth_side_trunk(pred_mask, cmin, cmax):
    bbox_w = cmax - cmin

    # Filter to body rows only (exclude head/leg thin rows)
    row_widths = np.sum(pred_mask[:, cmin:cmax+1], axis=1)
    max_width = row_widths.max()
    if max_width == 0:
        return None, None, None

    body_rows = np.where(row_widths > max_width * 0.70)[0]
    if len(body_rows) < 10:
        return None, None, None

    body_rmin, body_rmax = body_rows[0], body_rows[-1]
    body_mask = pred_mask[body_rmin:body_rmax+1, cmin:cmax+1]

    # Girth at shoulder, within body region only
    girth_col = int(bbox_w * 0.33)
    trunk_top, trunk_bot = get_body_trunk_at_col(body_mask, girth_col)

    if trunk_top is None:
        return None, None, None

    return int(trunk_bot - trunk_top + 1), body_rmin + trunk_top, body_rmin + trunk_bot

def get_largest_contour(mask):
    """Find largest contour in binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def extract_schaeffer_features(img_path, model, ann_side_path=None, ann_back_path=None):
    img_bgr   = cv2.imread(str(img_path))
    H, W      = img_bgr.shape[:2]
    px_per_cm = get_px_per_cm(ann_side_path)

    result = model.predict(str(img_path), verbose=False)[0]
    if result.masks is None or len(result.masks) == 0:
        return None

    best_idx  = int(result.boxes.conf.argmax())
    mask_data = result.masks.data[best_idx].cpu().numpy()
    pred_mask = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    cattle_px = int(pred_mask.sum())
    if cattle_px < 500:
        return None

    rows_ = np.any(pred_mask, axis=1); cols_ = np.any(pred_mask, axis=0)
    rmin, rmax = np.where(rows_)[0][[0, -1]]
    cmin, cmax = np.where(cols_)[0][[0, -1]]
    bbox_h = int(rmax - rmin + 1)
    bbox_w = int(cmax - cmin + 1)

    # Measurements
    length_px,  pt_A,     pt_B      = get_body_length_diagonal(pred_mask, cmin, cmax)
    girth_h_px, trunk_top, trunk_bot = get_girth_side_trunk(pred_mask, cmin, cmax)
    #girth_arc_px                     = get_girth_back_arc(ann_back_path)

    # Solidity
    contour   = get_largest_contour(pred_mask)
    hull_area = cv2.contourArea(cv2.convexHull(contour)) if contour is not None else 0
    solidity  = cattle_px / hull_area if hull_area > 0 else 0.0

    # Schaeffer — both girth methods, pick best available
    schaeffer_side_kg = schaeffer_arc_kg = None

    if px_per_cm and length_px:
        length_cm   = length_px / px_per_cm
        length_inch = length_cm * CM_TO_INCH

        if girth_h_px:
            # M1: treat body trunk height as half-diameter → half-circumference × 2
            girth_side_cm = (girth_h_px / px_per_cm) * np.pi
            schaeffer_side_kg = ((girth_side_cm * CM_TO_INCH)**2 * length_inch / 300) * LBS_TO_KG

    schaeffer_kg = schaeffer_side_kg

    return {
        'cattle_px'         : cattle_px,
        'cattle_area_norm'  : cattle_px / (W * H),
        'bbox_w'            : bbox_w,
        'bbox_h'            : bbox_h,
        'length_px_diag'    : length_px   or 0,
        'girth_h_px_trunk'  : girth_h_px  or 0,
        #'girth_arc_px'      : girth_arc_px or 0,
        'bbox_w_norm'       : bbox_w / W,
        'bbox_h_norm'       : bbox_h / H,
        'aspect_ratio'      : bbox_w / bbox_h if bbox_h > 0 else 0,
        'solidity'          : solidity,
        'schaeffer_side_kg' : schaeffer_side_kg,
        'schaeffer_arc_kg'  : schaeffer_arc_kg,
        'schaeffer_kg'      : schaeffer_kg,
        'has_sticker_scale' : 1 if px_per_cm else 0,
        #'has_back_view'     : 1 if girth_arc_px else 0,
        '_pt_A'             : pt_A,
        '_pt_B'             : pt_B,
        '_trunk_top'        : trunk_top,
        '_trunk_bot'        : trunk_bot,
    }

def extract_features(pil_image: Image.Image) -> dict:
    """Convert PIL image → temp file → extract_schaeffer_features."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        pil_image.save(f.name)
        tmp_path = f.name
    
    try:
        feats = extract_schaeffer_features(tmp_path, seg_model, None, None)
        return feats if feats else {}
    finally:
        os.unlink(tmp_path)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No photo uploaded!"}), 400

        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        features = extract_features(pil_img)
        
        # Map to training features
        features_mapped = {
            'cattle_px': features.get('cattle_px', 0),
            'cattle_area_norm': features.get('cattle_area_norm', 0),
            'bbox_w': features.get('bbox_w', 0),
            'bbox_h': features.get('bbox_h', 0),
            'girth_h_px': features.get('girth_h_px_trunk', 0),
            'back_w_px': features.get('back_w_px', 0),
            'bbox_w_norm': features.get('bbox_w_norm', 0),
            'bbox_h_norm': features.get('bbox_h_norm', 0),
            'aspect_ratio': features.get('aspect_ratio', 0),
            'solidity': features.get('solidity', 0),
            'schaeffer_kg': features.get('schaeffer_kg', 0),
        }
        
        X = pd.DataFrame([features_mapped])
        pred_log = pipeline.predict(X)[0]
        pred_kg = float(np.exp(pred_log))
        
        if pred_kg < 120: verdict = "Young calf! 🌱"
        elif pred_kg < 200: verdict = "Healthy young cow! 🐄"
        elif pred_kg < 300: verdict = "Fine mature cow! 💪"
        elif pred_kg < 400: verdict = "Big well-fed cow! 🌟"
        else: verdict = "Champion cow! 🏆"
        
        return jsonify({"weight_kg": round(pred_kg, 1), "verdict": verdict})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500






        X = pd.DataFrame([features])

        # Align columns to whatever the pipeline was trained with
        expected = getattr(pipeline[0], "feature_names_in_", None)
        if expected is not None:
            for col in expected:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[list(expected)]

        pred_log = pipeline.predict(X)[0]
        pred_kg  = float(np.exp(pred_log))

        if   pred_kg < 120: verdict = "A young calf — keep feeding her well! 🌱"
        elif pred_kg < 200: verdict = "A healthy young cow! 🐄"
        elif pred_kg < 300: verdict = "A fine mature cow! 💪"
        elif pred_kg < 400: verdict = "A big, well-fed cow! 🌟"
        else:               verdict = "Whoa! That's a champion cow! 🏆"

        return jsonify({"weight_kg": round(pred_kg, 1), "verdict": verdict})

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
features = extract_features(pil_img)
print(f"Features: {features}")  # see if they vary per image

X = pd.DataFrame([features])
pred_log = pipeline.predict(X)[0]
print(f"Pred log: {pred_log}")  # see if this changes