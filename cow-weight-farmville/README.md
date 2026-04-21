---
title: Cowville
emoji: 🐄
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🐄 Cowville — AI Cattle Weight Scale

A Farmville-themed web demo for a BSc final-year project on cattle weight
estimation from a single photograph.

## How it works

1. **YOLOv8 backbone** (fine-tuned on cattle body parts) extracts 640
   feature-maps from an uploaded cow photo. Global-average-pooled at layers
   P3/P4/P5 to produce a fixed-length visual feature vector.
2. Visual features are concatenated with tabular metadata (age, sex, breed).
3. A **Ridge / Gradient-Boosting ensemble** predicts log-weight; the output
   is exponentiated to kilograms.

Trained on the Pakistan BMGF cattle dataset + a custom SKU dataset.
**R² ≈ 0.83** on held-out cattle with typical error ≈ ±15 kg.

## Local run

```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:7860
```

Environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_PATH`    | `model/cow_weight_model.pkl`        | Joblib pipeline |
| `YOLO_WEIGHTS`  | `model/yolov12_cowparts_best.pt`    | YOLO weights (falls back to `yolov8n.pt` if missing) |
| `PORT`          | `7860`                              | HTTP port |

## Deploy to Hugging Face Spaces

1. Create a new Space → **Docker** SDK.
2. `git push` this folder to the Space repo.
3. Upload the model weights (`cow_weight_model.pkl`,
   `yolov12_cowparts_best.pt`) into `model/` via the HF web UI, or commit
   them via `git-lfs`.
4. Wait for the build to finish; the app appears at
   `https://huggingface.co/spaces/<user>/<space>`.

## Stack

- **Backend:** Flask + Gunicorn
- **ML:** PyTorch, Ultralytics YOLOv8, scikit-learn
- **Frontend:** vanilla HTML/CSS/JS (no framework) — Farmville-themed with
  a bouncing barn, drifting clouds, hopping livestock, and an animated
  weight-scale needle.

## Project structure

```
cow-weight-farmville/
├── app.py                      # Flask backend
├── requirements.txt
├── Dockerfile
├── model/                      # (gitignored) trained weights
│   ├── cow_weight_model.pkl
│   └── yolov12_cowparts_best.pt
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

🎓 *Final-year project · University of the West of England · 2025/26*
