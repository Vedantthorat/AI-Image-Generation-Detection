# AI Image Classification System
### Built with Python · TensorFlow · Keras · CNN

---

## Project Overview
An end-to-end image classification system using a Convolutional Neural Network (CNN)
trained on the CIFAR-10 dataset (60,000 images, 10 classes).

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## Project Structure
```
image_classification/
├── train.py           ← Train the CNN model
├── predict.py         ← Classify a new image
├── requirements.txt   ← Python dependencies
└── README.md
```

---

## Setup & Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python train.py
```
This will:
- Download CIFAR-10 automatically
- Preprocess and augment data
- Train the CNN (EarlyStopping enabled)
- Save best model as `best_model.keras`
- Save `results.png` with accuracy, loss, and confusion matrix

### Step 3: Predict on an image
```bash
# Predict on your own image
python predict.py --image your_image.jpg

# Demo using a random test image (no file needed)
python predict.py
```

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
Output: Dense(10, softmax)
```

---

## Expected Results
| Metric   | Value       |
|----------|-------------|
| Test Accuracy | ~88–92% |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | Up to 30 (EarlyStopping) |

---

## Key Concepts (for interview)

| Concept | What it means |
|---|---|
| CNN | Extracts spatial features using sliding filters |
| BatchNorm | Normalizes layer outputs for stable training |
| Dropout | Randomly deactivates neurons to prevent overfitting |
| Data Augmentation | Creates variations of training images artificially |
| EarlyStopping | Stops training when validation accuracy stops improving |
| Confusion Matrix | Shows which classes the model confuses with each other |

---

*Built by Vedant Thorat — B.Tech CSE (AI), NCER Pune*
