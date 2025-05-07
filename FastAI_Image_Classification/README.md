**Bird Species Classification with FastAI (CUB-200-2011)**

This repository contains a complete pipeline for training a deep learning model to classify **200 different bird species** using the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset. Built with the **FastAI** library, this project covers everything from data preprocessing and training to evaluation, visualization, and even testing adversarial inputs.

---

## 📌 Overview

- **Dataset**: CUB-200-2011 (200 bird species)
- **Model**: Pretrained ResNet architecture via FastAI's `cnn_learner`
- **Framework**: FastAI + PyTorch
- **Evaluation**: Accuracy, Error Rate, Confusion Matrix
- **Special Feature**: Basic adversarial overlay testing using `PIL`

---

## Dataset

The [CUB-200-2011 dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) 
contains:
- ~11,788 labeled images
- 200 distinct bird species
- Folder-structured images: `images/<class_folder>/<image>.jpg`

Make sure you download and place the dataset in the appropriate folder before running the notebook.

---

## Model Training Pipeline

### 🔹 Step 1: Data Preparation
- Used FastAI's `ImageDataLoaders.from_folder()` to automatically label images based on folder names.
- Applied basic transforms like `Resize(224)` and optional `aug_transforms`.

### 🔹 Step 2: Model Creation
- Created a CNN classifier using `cnn_learner()` with `resnet50`.
- Leveraged transfer learning with pretrained weights.

### 🔹 Step 3: Training
- Fine-tuned for 10 epochs using FastAI’s `fine_tune()` method.
- Metrics used: `accuracy`, `error_rate`.

### 🔹 Step 4: Evaluation
- Plotted a **confusion matrix** with FastAI and `seaborn.heatmap`.
- Identified top misclassifications using `.most_confused()`.
- Extracted prediction confidence values using `learn.predict(img)`.

---

## Prediction Confidence

To deepen the interpretability, the notebook displays:
- The top predicted class for each image
- The corresponding prediction confidence (e.g., `probs.max()`)

This helps identify low-confidence predictions and borderline cases.

---

## Adversarial Testing (Overlay)

A small experiment was conducted to simulate adversarial inputs:
- A semi-transparent white box was overlaid on top of bird images using Python’s `PIL` library.
- These altered images were used to test how subtle visual distortions affect model predictions.

---

## Inference on Single Images

You can run predictions on individual images like so:

```python
img = PILImage.create("path_to_image.jpg")
pred_class, pred_idx, probs = learn.predict(img)
print(f"Prediction: {pred_class}, Confidence: {probs.max().item():.2f}")


# Requirements
Python 3.8+
fastai ≥ 2.7
matplotlib, seaborn
Pillow (PIL)
scikit-learn

Instalation: (pip install fastai matplotlib seaborn scikit-learn pillow)

# Optional Add-On: Class-Specific Analysis
To zoom in on model performance for a specific class, you can use this mini confusion matrix function:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_for_class(class_name, y_true, y_pred, class_labels):
    class_idx = class_labels.index(class_name)
    binary_y_true = [1 if y == class_idx else 0 for y in y_true]
    binary_y_pred = [1 if y == class_idx else 0 for y in y_pred]

    cm = confusion_matrix(binary_y_true, binary_y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Other", class_name],
                yticklabels=["Other", class_name])
    plt.title(f"Confusion Matrix for Class: {class_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


Reference: https://github.com/rcalix1/TransferLearning/tree/main/fastai
