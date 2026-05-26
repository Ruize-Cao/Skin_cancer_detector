"""Shared class labels and display text for training, prediction, and reports."""

CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "NV",
    "MEL",
    "VASC",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_DISPLAY_NAMES = {
    "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
    "BCC": "Basal cell carcinoma",
    "BKL": "Benign keratosis-like lesion",
    "DF": "Dermatofibroma",
    "NV": "Melanocytic nevus",
    "MEL": "Melanoma",
    "VASC": "Vascular lesion",
}

CLASS_EXPLANATIONS = {
    "AKIEC": "A precancerous or very early cancer-related skin lesion category.",
    "BCC": "A common type of skin cancer that usually grows slowly but needs medical evaluation.",
    "BKL": "A group of benign keratosis-like skin lesions.",
    "DF": "A usually benign firm skin lesion category.",
    "NV": "A mole-like melanocytic lesion category.",
    "MEL": "A melanoma category, which is clinically important and should be reviewed promptly.",
    "VASC": "A vascular skin lesion category.",
}


def display_class_name(label: str) -> str:
    """Return a human-readable class name for a HAM10000 label code."""
    return CLASS_DISPLAY_NAMES.get(label, label)
