import json
import os
import uuid
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.report_agent import generate_batch_markdown_report, generate_markdown_report, markdown_to_pdf_bytes


"""Lightweight frontend/proxy app for Render deployments."""

app = FastAPI(title="skin_cancer_detector_frontend")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", "20"))
MAX_ZIP_BYTES = int(os.getenv("MAX_ZIP_BYTES", str(100 * 1024 * 1024)))


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
        <label for="file">Skin lesion image(s) or zipped image folder</label>
        <input id="file" name="files" type="file" accept="image/*,.zip" multiple required />
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
      const response = await fetch("/predict-batch", { method: "POST", body: new FormData(form) });
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
    return forward_image_to_inference(
        image_bytes=image_bytes,
        filename=file.filename or "upload.jpg",
        file_content_type=file.content_type or "application/octet-stream",
        network=network,
    )


@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    network: Annotated[str, Form()] = "EfficientNetB3",
):
    """Process multiple images or zipped image folders in deterministic order."""
    image_items = []
    for upload in files:
        image_items.extend(await extract_upload_images(upload))

    if not image_items:
        raise HTTPException(status_code=400, detail="No supported image files were uploaded.")
    if len(image_items) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum batch size is {MAX_BATCH_IMAGES}.",
        )

    results = []
    for index, item in enumerate(image_items, start=1):
        data = forward_image_to_inference(
            image_bytes=item["image_bytes"],
            filename=item["filename"],
            file_content_type=item["content_type"],
            network=network,
        )
        data["case_number"] = index
        data["case_id"] = f"Image {index:03d}"
        data["filename"] = data.get("filename") or item["filename"]
        results.append(data)

    if len(results) == 1:
        return results[0]

    return {
        "batch": True,
        "total": len(results),
        "network": network,
        "results": results,
        "text_report": generate_batch_markdown_report(results),
    }


@app.post("/report/pdf")
def report_pdf(result: dict):
    """Generate the report PDF locally from the inference result."""
    if result.get("batch") and isinstance(result.get("results"), list):
        markdown = result.get("text_report") or generate_batch_markdown_report(result["results"])
        filename = "skin-lesion-ai-batch-report.pdf"
    else:
        markdown = result.get("text_report") or generate_markdown_report(result)
        filename = "skin-lesion-ai-report.pdf"
    return Response(
        content=markdown_to_pdf_bytes(markdown),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def forward_image_to_inference(
    image_bytes: bytes,
    filename: str,
    file_content_type: str,
    network: str,
) -> dict:
    """Forward one image to the configured inference API."""
    body, content_type = build_multipart_body(
        image_bytes=image_bytes,
        filename=filename,
        file_content_type=file_content_type,
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=502, detail=f"Inference API request failed: {exc}") from exc


async def extract_upload_images(upload: UploadFile) -> list[dict]:
    """Return image byte entries from an image upload or a zip archive."""
    filename = upload.filename or "upload"
    data = await upload.read()
    if filename.lower().endswith(".zip"):
        if len(data) > MAX_ZIP_BYTES:
            raise HTTPException(status_code=400, detail="Uploaded zip file is too large.")
        return extract_zip_images(data)

    if not is_supported_image_name(filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

    return [
        {
            "filename": filename,
            "content_type": upload.content_type or guess_image_content_type(filename),
            "image_bytes": data,
        }
    ]


def extract_zip_images(zip_bytes: bytes) -> list[dict]:
    """Extract supported images from a zip archive, sorted by archive filename."""
    try:
        archive = zipfile.ZipFile(BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip file.") from exc

    items = []
    with archive:
        names = sorted(
            name for name in archive.namelist()
            if not name.endswith("/") and not PurePosixPath(name.replace("\\", "/")).name.startswith(".") and is_supported_image_name(name)
        )
        for name in names:
            with archive.open(name) as image_file:
                items.append(
                    {
                        "filename": name,
                        "content_type": guess_image_content_type(name),
                        "image_bytes": image_file.read(),
                    }
                )
    return items


def is_supported_image_name(filename: str) -> bool:
    """Return True when a filename has a supported image extension."""
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower() in IMAGE_EXTENSIONS


def guess_image_content_type(filename: str) -> str:
    """Infer a simple image content type from the filename extension."""
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


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
