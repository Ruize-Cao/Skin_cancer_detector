import json
import os
import uuid
import urllib.error
import urllib.request
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.report_agent import generate_markdown_report, markdown_to_pdf_bytes


"""Lightweight frontend/proxy app for Render or Vercel deployments."""

app = FastAPI(title="skin_cancer_detector_frontend")


def get_inference_url() -> str:
    """Read the Hugging Face Space inference endpoint from the environment."""
    base_url = os.getenv("INFERENCE_API_URL") or os.getenv("SKIN_CANCER_INFERENCE_API_URL")
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="INFERENCE_API_URL is not configured.",
        )
    return base_url.rstrip("/") + "/predict"


@app.get("/", response_class=HTMLResponse)
def web_app():
    """Serve the browser UI while forwarding predictions to the inference API."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>skin_cancer_detector</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #17202a; }
    main { max-width: 820px; margin: 0 auto; padding: 28px 18px; }
    section { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; }
    input, button, select { font: inherit; }
    input, select { width: 100%; margin: 8px 0 12px; box-sizing: border-box; }
    button { border: 0; border-radius: 6px; background: #1769aa; color: white; padding: 10px 14px; cursor: pointer; }
    button:disabled { background: #8291a3; cursor: wait; }
    pre { white-space: pre-wrap; background: #101820; color: #f4f7fb; padding: 14px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
  <main>
    <h1>skin_cancer_detector</h1>
    <section>
      <form id="form">
        <label for="file">Skin lesion image</label>
        <input id="file" name="file" type="file" accept="image/*" required />
        <label for="network">Prediction network</label>
        <select id="network" name="network">
          <option value="EfficientNetB3">EfficientNetB3</option>
          <option value="ResNet50">ResNet50</option>
          <option value="DenseNet121">DenseNet121</option>
          <option value="Ensemble">Ensemble (all three networks)</option>
        </select>
        <button id="predict" type="submit">Predict</button>
        <button id="pdf" type="button" disabled>Download PDF</button>
      </form>
      <pre id="output">Prediction result will appear here.</pre>
    </section>
  </main>
  <script>
    const form = document.getElementById("form");
    const output = document.getElementById("output");
    const predict = document.getElementById("predict");
    const pdf = document.getElementById("pdf");
    let latest = null;

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      predict.disabled = true;
      pdf.disabled = true;
      output.textContent = "Running prediction...";
      const response = await fetch("/predict", { method: "POST", body: new FormData(form) });
      latest = await response.json();
      output.textContent = JSON.stringify(latest, null, 2);
      predict.disabled = false;
      pdf.disabled = !response.ok;
    });

    pdf.addEventListener("click", async () => {
      if (!latest) return;
      const response = await fetch("/report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(latest)
      });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "skin-lesion-ai-report.pdf";
      link.click();
      URL.revokeObjectURL(url);
    });
  </script>
</body>
</html>
"""


@app.get("/health")
def health_check():
    """Return frontend state without importing TensorFlow or loading models."""
    return {
        "status": "ok",
        "mode": "frontend_proxy",
        "inference_api_configured": bool(
            os.getenv("INFERENCE_API_URL") or os.getenv("SKIN_CANCER_INFERENCE_API_URL")
        ),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    network: Annotated[str, Form()] = "EfficientNetB3",
):
    """Forward the upload to the Hugging Face Space inference API."""
    image_bytes = await file.read()
    body, content_type = build_multipart_body(
        image_bytes=image_bytes,
        filename=file.filename or "upload.jpg",
        file_content_type=file.content_type or "application/octet-stream",
        network=network,
    )
    request = urllib.request.Request(
        get_inference_url(),
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Inference API request failed: {exc}") from exc

    return data


@app.post("/report/pdf")
def report_pdf(result: dict):
    """Generate the report PDF locally from the inference result."""
    markdown = result.get("text_report") or generate_markdown_report(result)
    return Response(
        content=markdown_to_pdf_bytes(markdown),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=skin-lesion-ai-report.pdf"},
    )


def build_multipart_body(
    image_bytes: bytes,
    filename: str,
    file_content_type: str,
    network: str,
) -> tuple[bytes, str]:
    """Build a multipart/form-data payload without adding an HTTP dependency."""
    boundary = f"----skin-cancer-{uuid.uuid4().hex}"
    chunks = [
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="network"\r\n\r\n',
        network.encode("utf-8"),
        b"\r\n",
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {file_content_type}\r\n\r\n"
        ).encode("utf-8"),
        image_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
