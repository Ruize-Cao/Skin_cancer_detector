from __future__ import annotations

import re
import textwrap
from typing import Any

from training.constants import CLASS_EXPLANATIONS, display_class_name


"""Build human-readable Markdown and PDF reports from prediction results."""

DISCLAIMER = (
    "Educational/research output only. This prediction is not a medical "
    "diagnosis. Consult a qualified clinician for medical advice."
)
IMPORTANT_CLASSES = {"AKIEC", "BCC", "MEL"}


def generate_markdown_report(result: dict[str, Any]) -> str:
    """Create the text report returned by the API and embedded in the PDF."""
    prediction = str(result.get("prediction", "Unknown"))
    confidence = _as_float(result.get("confidence"))
    probabilities = result.get("probabilities") if isinstance(result.get("probabilities"), dict) else {}
    image = result.get("image") if isinstance(result.get("image"), dict) else {}

    lines = [
        "# Skin Lesion AI Report",
        "",
        "## Uploaded Image",
        _image_text(image),
        "",
        "## Prediction",
        f"- Predicted class: {display_class_name(prediction)}",
        f"- Model class code: {prediction}",
        f"- Confidence: {_percent(confidence)}",
        f"- Class note: {CLASS_EXPLANATIONS.get(prediction, 'Unknown class.')}",
        "",
        "## Result Summary",
        _summary(prediction, confidence),
        "",
        "## Image Feature Extraction",
        _feature_text(image, prediction),
        "",
        "## VLM Image Description",
        _vlm_text(result),
        "",
        "## Class Probabilities",
        "| Class | Probability |",
        "| --- | ---: |",
    ]

    for label, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {display_class_name(str(label))} ({label}) | {_percent(_as_float(score))} |")

    lines.extend(["", "## Disclaimer", result.get("disclaimer") or DISCLAIMER])
    return "\n".join(lines).strip() + "\n"


def generate_batch_markdown_report(results: list[dict[str, Any]]) -> str:
    """Create one ordered report for multiple image prediction results."""
    lines = [
        "# Skin Lesion AI Batch Report",
        "",
        f"Total images: {len(results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        case_id = result.get("case_id") or f"Image {index:03d}"
        filename = result.get("filename") or "uploaded image"
        prediction = str(result.get("prediction", "Unknown"))
        confidence = _as_float(result.get("confidence"))
        probabilities = result.get("probabilities") if isinstance(result.get("probabilities"), dict) else {}
        image = result.get("image") if isinstance(result.get("image"), dict) else {}

        lines.extend(
            [
                f"## {case_id} - {filename}",
                "",
                "### Uploaded Image",
                _image_text(image),
                "",
                "### Prediction",
                f"- Predicted class: {display_class_name(prediction)}",
                f"- Model class code: {prediction}",
                f"- Confidence: {_percent(confidence)}",
                f"- Class note: {CLASS_EXPLANATIONS.get(prediction, 'Unknown class.')}",
                "",
                "### Result Summary",
                _summary(prediction, confidence),
                "",
                "### Image Feature Extraction",
                _feature_text(image, prediction),
                "",
                "### VLM Image Description",
                _vlm_text(result),
                "",
                "### Class Probabilities",
                "| Class | Probability |",
                "| --- | ---: |",
            ]
        )

        for label, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| {display_class_name(str(label))} ({label}) | {_percent(_as_float(score))} |")
        lines.append("")

    lines.extend(["## Disclaimer", DISCLAIMER])
    return "\n".join(lines).strip() + "\n"


def markdown_to_pdf_bytes(markdown: str) -> bytes:
    """Render Markdown-ish text to a small self-contained PDF byte string."""
    lines = _plain_lines(markdown)
    page_lines = 48
    pages = [lines[i : i + page_lines] for i in range(0, len(lines), page_lines)] or [[""]]
    objects: list[bytes] = []

    _obj(objects, b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_index = _obj(objects, b"")
    font_id = _obj(objects, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []

    for page in pages:
        content_id = len(objects) + 2
        page_id = _obj(
            objects,
            (
                f"<< /Type /Page /Parent {pages_index} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii"),
        )
        page_ids.append(page_id)
        stream = _text_stream(page)
        _obj(objects, b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    objects[pages_index - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
        f"/Count {len(page_ids)} >>"
    ).encode("ascii")
    return _pdf(objects)


def _image_text(image: dict[str, Any]) -> str:
    if image.get("description"):
        return str(image["description"])

    filename = image.get("filename") or "uploaded image"
    content_type = image.get("content_type") or "unknown file type"
    width = image.get("width")
    height = image.get("height")
    if width and height:
        return f"The uploaded file is {filename} ({content_type}), {width}x{height} pixels."
    return f"The uploaded file is {filename} ({content_type})."


def _feature_text(image: dict[str, Any], prediction: str) -> str:
    """Explain the image cues and CNN feature extraction in plain English."""
    features = image.get("visual_features") if isinstance(image, dict) else None
    if not isinstance(features, dict):
        return (
            "The report did not receive extracted image features. The model still "
            "classified the image using learned convolutional feature representations."
        )

    return (
        f"Before prediction, the image is resized and passed through a convolutional "
        f"neural network. The network learns feature maps for color, texture, edges, "
        f"shape, and higher-level lesion patterns. A lightweight explanatory summary "
        f"computed from the uploaded pixels found: {features.get('summary', 'N/A')} "
        f"These cues help describe why the prediction was assigned to "
        f"{display_class_name(prediction)}, but they are not a clinician's diagnosis "
        f"and do not expose every internal feature used by the model."
    )


def _vlm_text(result: dict[str, Any]) -> str:
    """Summarize optional VLM observations or explain why VLM was skipped."""
    vlm = result.get("vlm_analysis")
    if not isinstance(vlm, dict):
        return "No VLM analysis was attached to this prediction."
    if vlm.get("available") and vlm.get("description"):
        return (
            f"Vision-language model observation ({vlm.get('provider', 'VLM')}"
            f"{' / ' + vlm['model'] if vlm.get('model') else ''}): "
            f"{vlm['description']} This is a visual description only, not a diagnosis."
        )
    if "disabled" in str(vlm.get("reason", "")).lower():
        return (
            "VLM image description is disabled for this cloud deployment. "
            "The report still includes CNN prediction results and local image feature extraction."
        )
    return (
        "VLM analysis was not run for this request. "
        f"Reason: {vlm.get('reason', 'not available')}"
    )


def _summary(prediction: str, confidence: float | None) -> str:
    text = f"The model predicts {display_class_name(prediction)} with confidence {_percent(confidence)}."
    if prediction in IMPORTANT_CLASSES:
        return text + " This class can be clinically important and should be reviewed by a qualified clinician."
    if confidence is not None and confidence < 0.6:
        return text + " Confidence is limited, so treat this result as uncertain."
    return text + " Clinical review is still required."


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _plain_lines(markdown: str) -> list[str]:
    lines = []
    for raw in markdown.splitlines():
        text = re.sub(r"^#{1,6}\s+", "", raw.strip()).replace("|", " ")
        text = re.sub(r"\s+", " ", text).strip()
        lines.extend(textwrap.wrap(text, width=88) or [""])
    return lines


def _text_stream(lines: list[str]) -> bytes:
    chunks = ["BT", "/F1 10 Tf", "54 744 Td"]
    for index, line in enumerate(lines):
        if index:
            chunks.append("0 -14 Td")
        chunks.append(f"({_escape(line)}) Tj")
    chunks.append("ET")
    return "\n".join(chunks).encode("latin-1", errors="replace")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _obj(objects: list[bytes], payload: bytes) -> int:
    objects.append(payload)
    return len(objects)


def _pdf(objects: list[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")

    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)
