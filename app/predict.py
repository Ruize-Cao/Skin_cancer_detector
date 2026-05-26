import os
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

from app.report_agent import DISCLAIMER
from training.constants import CLASS_NAMES


"""Prediction utilities shared by the API and tests."""

# Model path rules for the three selectable prediction networks.
NETWORKS = {
    "EfficientNetB3": {
        "env": "SKIN_CANCER_EFFICIENTNET_MODEL_PATH",
        "url_env": "SKIN_CANCER_EFFICIENTNET_MODEL_URL",
        "path": Path("Training_Sets") / "EfficientNetB3" / "T_next" / "best_model.keras",
        "fallback_path": Path("Training_Sets") / "EfficientNetB3" / "T7" / "best_model.keras",
    },
    "ResNet50": {
        "env": "SKIN_CANCER_RESNET_MODEL_PATH",
        "url_env": "SKIN_CANCER_RESNET_MODEL_URL",
        "path": Path("Training_Sets") / "ResNet50" / "R_next" / "best_model.keras",
        "fallback_path": Path("Training_Sets") / "ResNet50" / "R1" / "best_model.keras",
    },
    "DenseNet121": {
        "env": "SKIN_CANCER_DENSENET_MODEL_PATH",
        "url_env": "SKIN_CANCER_DENSENET_MODEL_URL",
        "path": Path("Training_Sets") / "DenseNet121" / "D_next" / "best_model.keras",
        "fallback_path": Path("Training_Sets") / "DenseNet121" / "D1" / "best_model.keras",
    },
}
ENSEMBLE_NETWORK = "Ensemble"


def normalize_network(network: str | None) -> str:
    """Return a supported network name, using EfficientNetB3 as the default."""
    if not network:
        return "EfficientNetB3"
    if network != ENSEMBLE_NETWORK and network not in NETWORKS:
        raise ValueError(f"Unsupported network: {network}")
    return network


def get_available_networks() -> list[str]:
    """List network names exposed to the web UI and prediction API."""
    return [*NETWORKS, ENSEMBLE_NETWORK]


def get_model_path(network: str | None = None) -> Path:
    """Resolve the model file path for a selected network."""
    network = normalize_network(network)
    if network == ENSEMBLE_NETWORK:
        return Path("Training_Sets") / "Ensemble"

    config = NETWORKS[network]

    env_path = os.getenv(config["env"])
    if env_path:
        return Path(env_path)

    default_env_path = os.getenv("SKIN_CANCER_MODEL_PATH")
    if default_env_path and network == "EfficientNetB3":
        return Path(default_env_path)

    if config["path"].exists():
        return config["path"]
    return config["fallback_path"]


def load_prediction_model(model_path: Path | None = None, network: str | None = None):
    """Load a saved Keras model for inference without recompiling it."""
    network = normalize_network(network)
    if network == ENSEMBLE_NETWORK:
        models = {
            name: load_prediction_model(get_model_path(name), name)
            for name in NETWORKS
        }
        if any(model is None for model in models.values()):
            return None
        return models

    path = model_path or get_model_path(network)
    ensure_model_file(path, network)
    if not is_valid_keras_file(path):
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except (OSError, ValueError):
        return None


def ensure_model_file(path: Path, network: str) -> None:
    """Download a missing model from an external URL configured in the environment."""
    if is_valid_keras_file(path) or network == ENSEMBLE_NETWORK:
        return

    url = os.getenv(NETWORKS[network]["url_env"])
    if not url:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(temp_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def is_valid_keras_file(path: Path) -> bool:
    """Check that a model path points to a real Keras zip file, not an LFS pointer."""
    return path.is_file() and zipfile.is_zipfile(path)


def preprocess_image(image_bytes: bytes, img_size: int):
    """Decode image bytes and resize them to the model input size."""
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)


def describe_uploaded_image(image_bytes: bytes, filename: str | None, content_type: str | None) -> dict:
    """Collect basic image metadata and lightweight visual feature summaries."""
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    height, width, channels = image.shape
    image_float = tf.cast(image, tf.float32) / 255.0
    brightness = float(tf.reduce_mean(image_float).numpy())
    contrast = float(tf.math.reduce_std(image_float).numpy())
    features = extract_visual_features(image_float)
    return {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": len(image_bytes),
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "visual_features": features,
        "description": (
            f"The uploaded file is {filename or 'uploaded image'} ({content_type or 'unknown file type'}), "
            f"{int(width)}x{int(height)} pixels. Basic image statistics suggest "
            f"{_brightness_label(brightness)} lighting with {_contrast_label(contrast)} contrast."
        ),
    }


def extract_visual_features(image_float: tf.Tensor) -> dict:
    """Compute simple visual cues used to explain the prediction report."""
    resized = tf.image.resize(image_float, [128, 128])
    red, green, blue = resized[:, :, 0], resized[:, :, 1], resized[:, :, 2]
    gray = tf.image.rgb_to_grayscale(resized)[:, :, 0]

    redness = float(tf.reduce_mean(red - (green + blue) / 2.0).numpy())
    color_variation = float(tf.reduce_mean(tf.math.reduce_std(resized, axis=[0, 1])).numpy())
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    edge_strength = float((tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))).numpy() / 2.0)
    blur = tf.nn.avg_pool2d(gray[None, :, :, None], ksize=9, strides=1, padding="SAME")[0, :, :, 0]
    texture_variation = float(tf.reduce_mean(tf.cast(tf.abs(gray - blur) > 0.12, tf.float32)).numpy())
    asymmetry = float(
        (
            tf.reduce_mean(tf.abs(gray[:, :64] - tf.reverse(gray[:, 64:], axis=[1])))
            + tf.reduce_mean(tf.abs(gray[:64, :] - tf.reverse(gray[64:, :], axis=[0])))
        ).numpy()
        / 2.0
    )

    features = {
        "dominant_color": _color_label(redness, float(tf.reduce_mean(gray).numpy())),
        "redness_score": round(redness, 4),
        "color_variation": round(color_variation, 4),
        "edge_strength": round(edge_strength, 4),
        "texture_variation": round(texture_variation, 4),
        "asymmetry": round(asymmetry, 4),
    }
    features["summary"] = (
        f"{features['dominant_color']} color, {_level(color_variation)} color variation, "
        f"{_level(edge_strength)} border/edge change, {_level(texture_variation)} texture variation, "
        f"and {_level(asymmetry)} asymmetry."
    )
    return features


def _brightness_label(value: float) -> str:
    if value < 0.33:
        return "dark"
    if value > 0.67:
        return "bright"
    return "moderate"


def _contrast_label(value: float) -> str:
    if value < 0.16:
        return "low"
    if value > 0.30:
        return "high"
    return "moderate"


def _color_label(redness: float, brightness: float) -> str:
    if redness > 0.08:
        return "reddish or pink-toned"
    if brightness < 0.35:
        return "dark brown or dark-toned"
    if brightness > 0.68:
        return "light-toned"
    return "brown or mixed-toned"


def _level(value: float) -> str:
    if value < 0.08:
        return "low"
    if value > 0.18:
        return "high"
    return "moderate"


def get_model_img_size(model) -> int:
    """Read the model input image size, falling back to the project default."""
    input_shape = getattr(model, "input_shape", None)
    if input_shape and input_shape[1]:
        return int(input_shape[1])
    return 300


def predict_image(model, image_bytes: bytes) -> dict:
    """Run inference and return the top class, confidence, and probabilities."""
    if isinstance(model, dict):
        probs = predict_ensemble(model, image_bytes)
    else:
        image = preprocess_image(image_bytes, get_model_img_size(model))
        probs = tf.nn.softmax(model(image, training=False), axis=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
        "disclaimer": DISCLAIMER,
    }


def predict_ensemble(models: dict[str, object], image_bytes: bytes) -> np.ndarray:
    """Average probability outputs from all selected CNN models."""
    all_probs = []
    for model in models.values():
        image = preprocess_image(image_bytes, get_model_img_size(model))
        probs = tf.nn.softmax(model(image, training=False), axis=1).numpy()[0]
        all_probs.append(probs)
    return np.mean(np.stack(all_probs), axis=0)
