"""
Mask Detection - Training Script
Uses MobileNetV2 transfer learning for high accuracy face mask detection.
Classes:  0 = with_mask  |  1 = without_mask  (alphabetical order by Keras)
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    AveragePooling2D, Dense, Dropout, Flatten, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # MobileNetV2 native input size
BATCH_SIZE  = 32
EPOCHS      = 20           # EarlyStopping will cut this short if needed
LR_INIT     = 1e-4
DATASET_DIR = "dataset"
MODEL_PATH  = "mask_detector.keras"

# ── Data generators (SEPARATE augmentation for train vs val) ─────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

# Validation: ONLY rescale — no augmentation!
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=42,
)

print("\nClass indices:", train_generator.class_indices)
# Expected: {'with_mask': 0, 'without_mask': 1}
# prediction < 0.5  →  with_mask
# prediction >= 0.5 →  without_mask

# ── Build model with MobileNetV2 backbone ────────────────────────────────────
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
)

# Freeze the base initially (feature extraction phase)
base_model.trainable = False

# Add custom classification head
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(1, activation="sigmoid")(head)

model = Model(inputs=base_model.input, outputs=head)

model.compile(
    optimizer=Adam(learning_rate=LR_INIT),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ── Phase 1: Train the head only ─────────────────────────────────────────────
print("\n[Phase 1] Training classification head only …")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
)

# ── Phase 2: Fine-tune the top layers of MobileNetV2 ─────────────────────────
print("\n[Phase 2] Fine-tuning top layers of MobileNetV2 …")

# Unfreeze the top 20 layers
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LR_INIT / 10),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training complete!")