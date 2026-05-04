"""
predict.py — Classify a new image using the trained model
==========================================================
Usage:
    python predict.py --image path/to/your/image.jpg

Or run without arguments for a quick demo on a random CIFAR-10 test image.
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
MODEL_PATH = "best_model.keras"
IMG_SIZE   = 32


def load_and_preprocess(image_path: str) -> np.ndarray:
    """Load an image from disk, resize to 32×32, normalize."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 32, 32, 3)


def predict_image(model, img_array: np.ndarray):
    """Return predicted class name and confidence."""
    probs      = model.predict(img_array, verbose=0)[0]
    class_idx  = np.argmax(probs)
    confidence = probs[class_idx] * 100
    return CLASS_NAMES[class_idx], confidence, probs


def show_result(img_array: np.ndarray, class_name: str, confidence: float, probs):
    """Display the image with prediction and probability bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: show image
    ax1.imshow(img_array[0])
    ax1.set_title(f"Prediction: {class_name}\nConfidence: {confidence:.1f}%",
                  fontsize=13, fontweight="bold", color="#1F4E79")
    ax1.axis("off")

    # Right: probability bar chart for all classes
    colors = ["#1F77B4" if c != class_name else "#FF7F0E" for c in CLASS_NAMES]
    ax2.barh(CLASS_NAMES, probs * 100, color=colors)
    ax2.set_xlabel("Probability (%)")
    ax2.set_title("Class Probabilities")
    ax2.axvline(50, color="gray", linestyle="--", alpha=0.4)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    print("  📊 Prediction chart saved to prediction_result.png")
    plt.show()


def demo_on_test_image(model):
    """Run prediction on a random CIFAR-10 test image (no file needed)."""
    print("  No image path provided — running demo on a random CIFAR-10 test image...\n")
    (_, _), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    idx        = np.random.randint(0, len(X_test))
    img_array  = (X_test[idx:idx+1].astype("float32") / 255.0)
    true_label = CLASS_NAMES[y_test[idx][0]]

    class_name, confidence, probs = predict_image(model, img_array)

    print(f"  True label  : {true_label}")
    print(f"  Predicted   : {class_name}  ({confidence:.1f}% confidence)")
    print(f"  Correct     : {'✅ Yes' if class_name == true_label else '❌ No'}")
    show_result(img_array, class_name, confidence, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image using the trained CNN.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file (jpg/png). Leave blank for demo.")
    args = parser.parse_args()

    print(f"\n  Loading model from '{MODEL_PATH}'...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("  Model loaded successfully.\n")
    except Exception as e:
        print(f"  ❌ Could not load model: {e}")
        print("  Run train.py first to train and save the model.")
        exit(1)

    if args.image:
        print(f"  Processing image: {args.image}")
        img_array              = load_and_preprocess(args.image)
        class_name, confidence, probs = predict_image(model, img_array)
        print(f"\n  Predicted class : {class_name}")
        print(f"  Confidence      : {confidence:.1f}%")
        show_result(img_array, class_name, confidence, probs)
    else:
        demo_on_test_image(model)
