# 🔍 VisionScan — AI Image Forensics

**Detect whether an image is AI-generated or a real photograph — entirely in your browser.**
<img width="1915" height="974" alt="image" src="https://github.com/user-attachments/assets/d0c4a9b6-e7fa-450e-bf23-65f22c1162fa" />
[![Live Demo](https://img.shields.io/badge/demo-live-22c55e)](https://aiimageclassifiervedant.netlify.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![TF.js](https://img.shields.io/badge/TensorFlow.js-in--browser-yellow)

---

## Overview

VisionScan is an end-to-end AI image detection system. A CNN trained on real vs. AI-generated images runs **entirely client-side via TensorFlow.js** — no server, no uploads, full privacy — and produces a detailed forensic report combining:

- **CNN classification** (AI probability score)
- **Pixel intensity histogram analysis** (RGB channel smoothness)
- **Frequency-domain fingerprinting** (GAN/diffusion artifact detection)
- **Texture & noise heuristics** (luminance variance, edge density, sensor noise)

The result is a full PDF-exportable forensic report, not just a single label.

---

## Demo

🔗 **Live app:** [aiimageclassifiervedant.netlify.app](https://aiimageclassifiervedant.netlify.app/)

---

## Project Structure

```
VisionScan/
│
├── README.md                      ← You are here
├── requirements.txt                ← Python dependencies
│
├── model/
│   ├── train.py                    ← Trains the CNN
│   ├── predict.py                  ← CLI prediction on a single image
│   ├── best_model.keras            ← Trained model weights
│   └── results.png                 ← Accuracy/loss curves + confusion matrix
│
├── tfjs_model/                     ← Browser-ready model (via tensorflowjs_converter)
│   ├── model.json
│   └── group1-shard1of1.bin
│
├── web/
│   ├── index.html                  ← VisionScan dashboard (upload, scan, report)
│   └── assets/
│       └── prediction_result.png
│
└── docs/
    └── forensic_methodology.md     ← Six-stage detection pipeline explained
```

---

## How It Works

| Stage | Description |
|---|---|
| 1. Preprocessing | Image resized to 32×32 (CNN input) / 64×64 (heuristic analysis), normalised to [0,1] |
| 2. CNN Feature Extraction | Three Conv2D blocks (BatchNorm + MaxPool + Dropout) extract hierarchical features |
| 3. Softmax Classification | Dense output head produces AI vs. Real probability |
| 4. Histogram Analysis | AI images show unnaturally smooth RGB histograms — real photos are jagged from sensor noise |
| 5. Frequency Fingerprinting | GAN/diffusion models leave periodic off-centre artifacts, visible in frequency space |
| 6. In-Browser Inference | Keras model converted via `tensorflowjs_converter`, runs fully client-side |

---

## Setup & Run (Model Training)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the model**
```bash
python model/train.py
```
This will:
- Load and preprocess the training dataset
- Train the CNN with EarlyStopping
- Save the best model as `model/best_model.keras`
- Save `model/results.png` with accuracy, loss, and confusion matrix

**3. Predict on a new image**
```bash
python model/predict.py --image your_image.jpg
```

**4. Convert model for browser use**
```bash
tensorflowjs_converter --input_format=keras model/best_model.keras tfjs_model/
```

**5. Run the web app**
Open `web/index.html` in a browser, or deploy the `web/` folder to Netlify/Vercel.

---

## Model Architecture

```
Input (32×32×3)
    ↓
Data Augmentation (flip, rotate, zoom, brightness)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.30)
    ↓
Conv Block 3: Conv2D(128) × 2 → BatchNorm → ReLU → MaxPool → Dropout(0.40)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.50)
    ↓
Output: Dense(2, softmax) — [Real, AI-Generated]
```

| Metric | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | Up to 30 (EarlyStopping) |

---

## Features

- 🖼️ Drag-and-drop image upload
- 📊 Live AI-probability gauge with confidence scoring
- 📈 Real-time pixel histogram (R/G/B channel analysis)
- 🌊 Frequency-domain fingerprint visualization
- 🔲 Texture & colour forensic breakdown
- 📄 One-click downloadable PDF forensic report
- 🕘 Scan history with per-image results
- 🌓 Dark / Blur theme toggle
- 🔒 100% client-side — no image data leaves the browser

---

## Tech Stack

**Model:** Python 3.10, TensorFlow 2.x, Keras
**Web:** HTML, CSS, JavaScript, TensorFlow.js
**Reporting:** jsPDF, jsPDF-AutoTable

---

## Disclaimer

VisionScan is a forensic *aid*, not a definitive legal determination. Results can be affected by compression, post-processing, and model limitations. For high-stakes verification, cross-check with dedicated tools (e.g. ExifTool, C2PA metadata).

---

## Author

**Vedant Thorat**
B.Tech CSE (AI Specialization) — PCET NCER, Pune · Batch 2023–27
[GitHub](https://github.com/Vedantthorat) · [Portfolio](https://vedantthoratportfolio1.netlify.app)
