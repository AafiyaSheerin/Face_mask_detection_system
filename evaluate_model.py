"""
evaluate_model.py
-----------------
Loads the saved mask_detector.keras model and evaluates it on the dataset.
Prints real accuracy, loss, precision, recall, F1-score, and a confusion matrix.

Classes (alphabetical order, as Keras assigns them):
  0 = with_mask
  1 = without_mask

Run with:
  python evaluate_model.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Config (must match train_model.py)
IMG_SIZE    = 224
BATCH_SIZE  = 32
DATASET_DIR = "dataset"
MODEL_PATH  = "mask_detector.keras"

SEP = "=" * 55
DIV = "-" * 40

# Load the saved model
print("\n" + SEP)
print("  FACE MASK DETECTOR - MODEL EVALUATION")
print(SEP)
print(f"\n[INFO] Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("[SUCCESS] Model loaded.\n")

# Create a clean test generator (NO augmentation, NO shuffle)
# Uses validation_split=0.2 with seed=42 to reproduce the exact same
# 20% validation split that was used during training.
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

test_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",   # same 20% held-out set used during training
    shuffle=False,         # IMPORTANT: must be False for accurate evaluation
    seed=42,
)

class_names = list(test_generator.class_indices.keys())
print(f"[INFO] Class mapping: {test_generator.class_indices}")
print(f"[INFO] Total images to evaluate: {test_generator.samples}\n")

# Evaluate using model.evaluate()
print("[INFO] Running model.evaluate() on test set...")
loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + DIV)
print(f"  Loss     : {loss:.4f}")
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(DIV + "\n")

# Get per-image predictions for detailed report
print("[INFO] Generating per-image predictions for detailed report...")
test_generator.reset()
y_pred_scores = model.predict(test_generator, verbose=1)

# Convert sigmoid scores to binary labels (same threshold as app.py)
y_pred = (y_pred_scores.flatten() >= 0.5).astype(int)
y_true = test_generator.classes  # ground truth labels

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n" + SEP)
print("  CONFUSION MATRIX")
print(SEP)
print(f"\n  Predicted ->    {class_names[0]:<15} {class_names[1]}")
print(f"  {class_names[0]:<16}   {cm[0][0]:<15} {cm[0][1]}")
print(f"  {class_names[1]:<16}   {cm[1][0]:<15} {cm[1][1]}")

TP = cm[0][0]  # with_mask predicted correctly
FN = cm[0][1]  # with_mask predicted as without_mask
FP = cm[1][0]  # without_mask predicted as with_mask
TN = cm[1][1]  # without_mask predicted correctly

print(f"\n  True Positives  (Mask -> Mask)       : {TP}")
print(f"  False Negatives (Mask -> No Mask)    : {FN}")
print(f"  False Positives (No Mask -> Mask)    : {FP}")
print(f"  True Negatives  (No Mask -> No Mask) : {TN}")

# Full Classification Report
print("\n" + SEP)
print("  FULL CLASSIFICATION REPORT")
print(SEP + "\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Final Summary
real_acc = accuracy_score(y_true, y_pred) * 100
print(SEP)
print(f"  FINAL REAL ACCURACY : {real_acc:.2f}%")
print(f"  FINAL REAL LOSS     : {loss:.4f}")
print(f"  IMAGES TESTED       : {test_generator.samples}")
print(SEP + "\n")
