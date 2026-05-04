"""
AI Image Classification System
================================
Uses CNN (Convolutional Neural Network) with TensorFlow/Keras.
Dataset: CIFAR-10 (10 classes, 60,000 images) — no manual download needed.

How to run:
    pip install tensorflow matplotlib scikit-learn
    python train.py
"""
# In your train.py — also save as TF.js format:
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "tfjs_model/")
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = 32          # CIFAR-10 images are 32x32
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS      = 30          # EarlyStopping will stop earlier if needed
MODEL_PATH  = "best_model.keras"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─────────────────────────────────────────────
# 2. DATA COLLECTION (load CIFAR-10 dataset)
# ─────────────────────────────────────────────
print("\n[1/6] Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2/6] Preprocessing data...")

# Normalize pixel values from [0, 255] → [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Create validation split (15% of training data)
val_split  = int(0.15 * len(X_train))
X_val      = X_train[:val_split]
y_val      = y_train[:val_split]
X_train    = X_train[val_split:]
y_train    = y_train[val_split:]

# One-hot encode labels  e.g. 3 → [0,0,0,1,0,0,0,0,0,0]
y_train_oh = to_categorical(y_train, NUM_CLASSES)
y_val_oh   = to_categorical(y_val,   NUM_CLASSES)
y_test_oh  = to_categorical(y_test,  NUM_CLASSES)

print(f"      Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 4. DATA AUGMENTATION
# ─────────────────────────────────────────────
print("\n[3/6] Setting up data augmentation pipeline...")

# Augmentation applied only during training to improve generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),          # Horizontal flip
    layers.RandomRotation(0.1),               # Rotate up to ±10%
    layers.RandomZoom(0.1),                   # Zoom in/out by ±10%
    layers.RandomTranslation(0.1, 0.1),       # Shift image slightly
    layers.RandomBrightness(0.1),             # Slight brightness change
], name="data_augmentation")

# ─────────────────────────────────────────────
# 5. MODEL BUILDING (CNN Architecture)
# ─────────────────────────────────────────────
print("\n[4/6] Building CNN model...")

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation (only active during training)
    x = data_augmentation(inputs)

    # Block 1: Conv → BatchNorm → ReLU → Pool
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2: Deeper filters
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3: Even deeper
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.4)(x)

    # Classifier Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="ImageClassifier_CNN")

model = build_model()
model.summary()

# Compile with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────────────────────────────────
# 6. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[5/6] Training model...")

callbacks = [
    # Stop training if val_accuracy doesn't improve for 5 epochs
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    # Save the best model checkpoint
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train_oh,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val_oh),
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating model...")

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n  ✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"  ✅ Test Loss     : {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_test.flatten()

# Classification report (Precision, Recall, F1 per class)
print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 8. PLOTS — Training History & Confusion Matrix
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("AI Image Classification — Results", fontsize=14, fontweight="bold")

# Plot 1: Accuracy curve
axes[0].plot(history.history["accuracy"],     label="Train Accuracy",  color="#1F77B4")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",    color="#FF7F0E")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Loss curve
axes[1].plot(history.history["loss"],     label="Train Loss", color="#2CA02C")
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#D62728")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[2])
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\n  📊 Results saved to results.png")
plt.show()

print("\n  💾 Best model saved to:", MODEL_PATH)
print("\n✅ Done! Run predict.py to classify your own images.")
