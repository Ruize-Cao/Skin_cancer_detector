from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

from app.predict import describe_uploaded_image, predict_image
from app.report_agent import DISCLAIMER, generate_markdown_report


"""Lightweight multi-agent pipeline for CNN prediction, optional VLM analysis, and report writing."""


class CNNPredictionAgent:
    """Runs the selected CNN model and returns class probabilities."""

    def run(self, model, image_bytes: bytes) -> dict[str, Any]:
        return predict_image(model, image_bytes)


class ImageFeatureAgent:
    """Extracts local image metadata and simple visual features."""

    def run(
        self,
        image_bytes: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> dict[str, Any]:
        return describe_uploaded_image(image_bytes, filename, content_type)


class VLMImageDescriptionAgent:
    """Asks local Ollama/LLaVA for a non-diagnostic image description when available."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OLLAMA_VLM_MODEL", "llava")
        self.url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        self.enabled = os.getenv("VLM_ENABLED", "true").lower() not in {"0", "false", "no"}

    def run(self, image_bytes: bytes, content_type: str | None) -> dict[str, Any]:
        if not self.enabled:
            return {
                "available": False,
                "provider": "ollama",
                "model": self.model,
                "reason": "VLM is disabled for this deployment.",
            }

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "prompt": (
                "Describe the visible skin-lesion image features for an educational AI report. "
                "Focus only on non-diagnostic visual observations such as color, border, shape, "
                "texture, asymmetry, lighting, and image quality. Do not give a medical diagnosis."
            ),
            "images": [image_b64],
            "stream": False,
        }

        try:
            request = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return {
                "available": False,
                "provider": "ollama",
                "model": self.model,
                "reason": str(exc),
            }

        return {
            "available": True,
            "provider": "ollama",
            "model": self.model,
            "description": str(data.get("response", "")).strip(),
        }


class SafetyAgent:
    """Ensures the final result always carries the educational medical disclaimer."""

    def run(self, result: dict[str, Any]) -> dict[str, Any]:
        result["disclaimer"] = result.get("disclaimer") or DISCLAIMER
        return result


class ReportWriterAgent:
    """Builds the final Markdown report from the combined agent outputs."""

    def run(self, result: dict[str, Any]) -> str:
        return generate_markdown_report(result)


class MultiAgentPredictionPipeline:
    """Coordinates prediction, image description, VLM analysis, safety, and report generation."""

    def __init__(self):
        self.cnn_agent = CNNPredictionAgent()
        self.image_agent = ImageFeatureAgent()
        self.vlm_agent = VLMImageDescriptionAgent()
        self.safety_agent = SafetyAgent()
        self.report_agent = ReportWriterAgent()

    def run(
        self,
        model,
        image_bytes: bytes,
        filename: str | None,
        content_type: str | None,
        network: str,
        model_path: str,
    ) -> dict[str, Any]:
        result = self.cnn_agent.run(model, image_bytes)
        result["image"] = self.image_agent.run(image_bytes, filename, content_type)
        result["vlm_analysis"] = self.vlm_agent.run(image_bytes, content_type)
        result["network"] = network
        result["model_path"] = model_path
        self.safety_agent.run(result)
        result["text_report"] = self.report_agent.run(result)
        return result
