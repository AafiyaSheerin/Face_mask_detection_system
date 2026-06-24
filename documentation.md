# 😷 Face Mask Detection System — Project Documentation

## 1. Project Overview

| Field | Details |
|---|---|
| **Project Title** | Real-Time Face Mask Detection System |
| **Objective** | Automatically detect whether a person is wearing a face mask using deep learning |
| **Type** | Binary Image Classification |
| **Interface** | Interactive Web Application (Streamlit) |
| **Final Accuracy** | **99.55% validation accuracy** |

### Problem Statement
During the COVID-19 pandemic and in safety-regulated environments, manually monitoring mask compliance is impractical. This system automates mask detection from images using AI, providing instant, accurate predictions.

---

## 2. Dataset

### Source & Composition

| Category | Number of Images |
|---|---|
| **With Mask** | 5,003 images |
| **Without Mask** | 5,001 images |
| **Total** | **~10,004 images** |

### Dataset Structure
```
dataset/
├── with_mask/        ← 5,003 face images wearing masks
│   ├── Augmented_0_xxx.png
│   ├── Augmented_1_xxx.png
│   └── ...
└── without_mask/     ← 5,001 face images without masks
    ├── Augmented_0_xxx.png
    └── ...
```

### Dataset Characteristics
- **Balanced**: Nearly equal images in both classes (avoids class bias)
- **Augmented**: Dataset contains pre-augmented images for diversity
- **Variety**: Includes different ethnicities, ages, lighting conditions, mask colors and styles (surgical, cloth, patterned, colored)
- **Format**: PNG/JPG images of faces

### Data Split
| Split | Percentage | Approx. Images |
|---|---|---|
| Training Set | 80% | ~8,003 |
| Validation Set | 20% | ~2,001 |

---

## 3. Models Used

### 3.1 Primary Model — MobileNetV2 (Transfer Learning)

> **Used for**: Mask classification (with mask / without mask)

| Property | Value |
|---|---|
| **Base Architecture** | MobileNetV2 |
| **Pre-trained On** | ImageNet (1.4 million images, 1000 classes) |
| **Input Size** | 224 × 224 × 3 (RGB) |
| **Output** | Single sigmoid neuron (binary: 0 = with mask, 1 = without mask) |
| **Total Parameters** | ~2.3 million |

#### Why MobileNetV2?
- Lightweight yet powerful — designed for mobile/embedded applications
- Uses **depthwise separable convolutions** → fast and efficient
- Pre-trained on ImageNet → already understands facial features, textures, shapes
- Excellent accuracy with minimal training data required

#### Architecture Diagram
```
Input Image (224×224×3)
        ↓
[MobileNetV2 Backbone — frozen initially]
  • 53 layers of depthwise separable convolutions
  • Inverted residual blocks
  • Progressive channel expansion (32 → 96 → 192 → 320)
        ↓
AveragePooling2D (7×7)
        ↓
Flatten
        ↓
Dense(128, activation='relu')
        ↓
Dropout(0.5)           ← prevents overfitting
        ↓
Dense(1, activation='sigmoid')   ← final prediction
        ↓
Output: probability score [0.0 → 1.0]
  • < 0.5 = WITH MASK ✅
  • ≥ 0.5 = WITHOUT MASK ❌
```

### 3.2 Face Detection — OpenCV Haar Cascade Classifiers

> **Used for**: Locating faces in the uploaded image before running mask classification

| Cascade | Purpose |
|---|---|
| `haarcascade_frontalface_default.xml` | Standard front-facing faces |
| `haarcascade_frontalface_alt2.xml` | Alternate frontal detection |
| `haarcascade_profileface.xml` | Side/profile faces |

#### How Haar Cascade Works
- Uses **Haar-like features** (edge, line, rectangle patterns) to detect face regions
- Trained using **AdaBoost algorithm** to select most discriminative features
- Uses a **cascade of classifiers** — fast rejection of non-face regions
- Runs on grayscale images with histogram equalization for better contrast

#### Smart Filtering Applied
- **Minimum size filter**: Only detects faces > 10% of image size (removes false positives like mask graphics/patterns)
- **Nested box removal**: Removes small boxes inside larger boxes
- **NMS (Non-Maximum Suppression)**: Removes duplicate overlapping detections (IoU > 40%)
- **Fallback**: If no face detected → smart crop + majority vote prediction

---

## 4. Training Methodology

### Two-Phase Transfer Learning

#### Phase 1 — Feature Extraction (Epochs 1–10)
```
• MobileNetV2 base: FROZEN (weights not updated)
• Only classification head is trained
• Learning rate: 1e-4
• Goal: Learn mask-specific features on top of ImageNet knowledge
```

#### Phase 2 — Fine-Tuning (Epochs 11–20)
```
• Top 20 layers of MobileNetV2: UNFROZEN
• Both head + top layers updated together
• Learning rate: 1e-5 (10× smaller — careful fine-tuning)
• Goal: Adapt base features specifically to mask detection
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 32 |
| Max Epochs | 20 (Phase 1: 10, Phase 2: 20) |
| Image Size | 224 × 224 |

### Data Augmentation (Training Only)
| Augmentation | Value | Purpose |
|---|---|---|
| Rotation | ±20° | Handle tilted photos |
| Zoom | 15% | Various distances |
| Width/Height Shift | 20% | Off-centre faces |
| Shear | 15% | Perspective variation |
| Horizontal Flip | Yes | Mirror images |

> **Note**: Validation set uses ONLY rescaling (no augmentation) to get true accuracy measurement.

### Callbacks Used
| Callback | Configuration | Purpose |
|---|---|---|
| `ModelCheckpoint` | Save best val_accuracy model | Keep only the best model |
| `EarlyStopping` | Patience = 5 epochs | Stop if no improvement |
| `ReduceLROnPlateau` | Factor=0.5, Patience=3 | Halve LR on plateau |

---

## 5. Results & Performance

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **99.55%** |
| **Final Validation Accuracy** | 99.40% |
| **Final Validation Loss** | 0.016 |
| **Training Accuracy** | ~99.43% |
| **Training Loss** | ~0.019 |
| **Best Epoch** | Epoch 5 |
| **Early Stopping Triggered** | Yes (Epoch 10 of Phase 2) |

### Comparison: Before vs After

| Aspect | Original Model | Improved Model |
|---|---|---|
| Architecture | Simple 3-layer CNN | MobileNetV2 Transfer Learning |
| Epochs | 10 | 20 (with early stopping) |
| Val Accuracy | ~75–80% (estimated) | **99.55%** |
| Flashy mask detection | ❌ Fails | ✅ Works |
| False positives | ❌ Many | ✅ Very few |
| Training time | ~5 min | ~2 hours (CPU) |

---

## 6. Application Architecture

```
User uploads image
        ↓
[Streamlit Web App — app.py]
        ↓
Face Detection (OpenCV Haar Cascade)
   ├── Face found → Crop face region (+10% padding)
   └── No face → Smart 3-crop majority vote fallback
        ↓
Preprocessing
   • Resize to 224×224
   • Normalize pixels to [0, 1]
   • Expand dims → (1, 224, 224, 3)
        ↓
MobileNetV2 Model Prediction
   • Output score ∈ [0.0, 1.0]
        ↓
Result Display
   • Score < 0.5 → ✅ MASK DETECTED
   • Score ≥ 0.5 → ❌ NO MASK DETECTED
   • Confidence % shown
   • Bounding boxes drawn on image
```

---

## 7. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10 |
| Deep Learning | TensorFlow / Keras | 2.21.0 |
| Pre-trained Model | MobileNetV2 (ImageNet) | via `tf.keras.applications` |
| Face Detection | OpenCV Haar Cascade | 4.8+ |
| Web Framework | Streamlit | 1.28+ |
| Image Processing | Pillow (PIL) | 10.0+ |
| Numerical Computing | NumPy | 1.24+ |

---

## 8. Key Concepts Explained

### Transfer Learning
Reusing a model trained on a large dataset (ImageNet) for a different but related task (mask detection). The model already knows faces, edges, textures — we just teach it the difference between masked and unmasked faces.

### Binary Classification
Two output classes:
- **Class 0**: `with_mask` (sigmoid output → 0.0)
- **Class 1**: `without_mask` (sigmoid output → 1.0)

### Sigmoid Activation
Output layer uses sigmoid: `σ(x) = 1 / (1 + e^(-x))`
Produces a probability between 0 and 1. Threshold at 0.5 for binary decision.

### Dropout Regularization
Randomly disables 50% of neurons during training → prevents overfitting → better generalization to new images.

---

## 9. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Whole-image false positives (scarf/lanyard) | Added face detection to crop face first |
| Haar Cascade failing on masked faces | Multi-cascade strategy + fallback prediction |
| Cartoon/patterned masks (Spider-Man) detected as multiple faces | Nested box removal + min size filter (10% of image) |
| Side-angle faces not detected | Profile cascade + flipped image detection |
| Old model failing on colorful masks | MobileNetV2 — understands face structure, not just color |

---

## 10. File Structure

```
Mask_Prediction_Project/
├── dataset/
│   ├── with_mask/        ← 5,003 training images
│   └── without_mask/     ← 5,001 training images
├── train_model.py        ← Model training script
├── app.py                ← Streamlit web application
├── mask_detector.keras   ← Trained model file (21 MB)
└── requirements.txt      ← Python dependencies
```

---

## 11. How to Run

### Train the Model
```bash
cd Mask_Prediction_Project
python train_model.py
```

### Run the Web App
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8501`

---

*Documentation prepared for Face Mask Detection Project Presentation*
