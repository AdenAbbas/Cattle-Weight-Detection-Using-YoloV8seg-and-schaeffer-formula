# Cattle Weight Estimation from RGB Images

**UFCEKP-30-3 — Data Science and AI Individual Project**

Estimating live cattle weight from a single side-profile smartphone photo, using instance segmentation and machine learning — no depth sensor, no scale required.

---

## Overview

This project investigates whether cattle live weight can be reliably predicted from standard RGB images alone. The pipeline combines:

1. **YOLOv8n-seg** fine-tuned on the Cattle Weight Detection Model Dataset (Kaggle, 12K images) to extract pixel-level body silhouettes
2. **Morphometric feature extraction** — body length (shoulder-to-pin-bone diagonal), estimated heart girth (trunk height at shoulder), bounding box ratios, and solidity
3. **Schaeffer formula** as a biomechanical prior, converting pixel measurements to a weight estimate
4. **Ridge regression** trained on a 12-feature vector with GroupKFold cross-validation to prevent animal-level data leakage

**Results:** R² = 0.410, MAE = 24.8 kg on the held-out test set.

---

## Repository Structure

```
├── Cow Weight Detection.ipynb      # Full training pipeline, EDA, and model evaluation
├── Final Draft.docx                # Final project report
├── cow-weight-farmville/           # Flask web application
│   ├── app.py                      # Backend — segmentation + feature extraction + prediction
│   ├── templates/index.html        # Frontend UI
│   ├── static/                     # CSS and JS
│   ├── model/
│   │   ├── cow_weight_model.pkl    # Trained Ridge regression pipeline
│   │   └── yolov8n_seg_cattle.pt  # Fine-tuned YOLOv8n-seg weights
│   ├── requirements.txt
│   └── Dockerfile
```

---

## Web Application

The Cowville app lets you upload a side-profile cow photo and receive an estimated weight instantly.

### Run locally

```bash
cd cow-weight-farmville
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860`.

### Run with Docker

```bash
cd cow-weight-farmville
docker build -t cowville .
docker run -p 7860:7860 cowville
```

---

## Notebook

`Cow Weight Detection.ipynb` covers the full pipeline:

- Dataset loading and exploratory analysis
- YOLOv8n-seg fine-tuning and mask extraction
- Morphometric feature engineering (body length, heart girth, solidity)
- Schaeffer formula implementation
- Ridge regression training with GroupKFold cross-validation
- SHAP feature importance analysis
- Comparison of alternative approaches (CNN backbone features, body-part segmentation)

---

## Dataset

**Cattle Weight Detection Model Dataset** — Kaggle (12K annotated images)  
Side-profile and rear-view images with pixel-level colour-coded masks (cattle body, calibration sticker, ground, background).

---

## Key Results

| Method | R² | MAE (kg) |
|---|---|---|
| Schaeffer formula (baseline) | −0.120 | 38.3 |
| Ridge regression (this project) | **0.410** | **24.8** |
| CNN backbone features | <0 | — |

---

## Tech Stack

- Python 3.10+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- scikit-learn · OpenCV · NumPy · pandas · Flask
