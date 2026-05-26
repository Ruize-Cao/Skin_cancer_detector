from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from training.constants import CLASS_NAMES


"""Evaluation utilities for plots, TTA predictions, and reports."""


def plot_training_curves(history, output_dir: Path):
    """Save accuracy and loss curves from a Keras training history."""

    output_dir.mkdir(parents=True, exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    # -------------------------
    # Accuracy Figure
    # -------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs_range,
        acc,
        label="Training Accuracy",
    )

    plt.plot(
        epochs_range,
        val_acc,
        label="Validation Accuracy",
    )

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=150,
    )

    plt.close()

    # -------------------------
    # Loss Figure
    # -------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs_range,
        loss,
        label="Training Loss",
    )

    plt.plot(
        epochs_range,
        val_loss,
        label="Validation Loss",
    )

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=150,
    )

    plt.close()

def tta_predict(model, images):
    """Average predictions over small test-time image transformations."""
    predictions = []

    # Original image
    predictions.append(model(images, training=False))

    # Horizontal flip
    flipped = tf.image.flip_left_right(images)
    predictions.append(model(flipped, training=False))

    # Slight brightness adjustment
    bright = tf.image.adjust_brightness(images, delta=0.05)
    predictions.append(model(bright, training=False))

    # Slight contrast adjustment
    contrast = tf.image.adjust_contrast(images, contrast_factor=1.1)
    predictions.append(model(contrast, training=False))

    # Average logits
    avg_logits = tf.reduce_mean(tf.stack(predictions), axis=0)

    return avg_logits

def apply_thresholds(prob):
    """
    Apply conservative threshold tuning for medically important classes.

    This avoids aggressively overriding the model's original argmax prediction.
    It only changes the prediction when a priority class is both confident enough
    and close to the model's original top prediction.
    """

    pred = int(np.argmax(prob))
    max_prob = prob[pred]

    priority_thresholds = {
        5: 0.45,  # MEL
        0: 0.40,  # AKIEC
        3: 0.40,  # DF
        6: 0.40,  # VASC
    }

    for class_idx, threshold in priority_thresholds.items():
        if prob[class_idx] >= threshold and prob[class_idx] >= max_prob * 0.85:
            return class_idx

    return pred

def evaluate_model(model, val_ds, output_dir: Path):
    """Write classification metrics and a confusion matrix for validation data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        logits = tta_predict(model, images)
        probs = tf.nn.softmax(logits, axis=1)
        
        preds = []

        for prob in probs.numpy():
            preds.append(apply_thresholds(prob))

        preds = np.array(preds)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(report)
