import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.agents import MultiAgentPredictionPipeline
from app.frontend import MAX_BATCH_IMAGES, extract_upload_images
from app.predict import (
    DISCLAIMER,
    NETWORKS,
    get_available_networks,
    get_model_path,
    load_prediction_model,
    normalize_network,
)
from app.report_agent import generate_batch_markdown_report, generate_markdown_report, markdown_to_pdf_bytes


"""FastAPI entrypoint for image upload, prediction, and PDF report download."""

app = FastAPI(title="skin_cancer_detector")
model_cache = {}
prediction_pipeline = MultiAgentPredictionPipeline()


def get_model(network: str | None = None):
    """Load and cache the selected Keras model for prediction requests."""
    network = normalize_network(network)
    model_path = get_model_path(network)
    cache_key = f"{network}:{model_path}"
    if cache_key not in model_cache:
        model_cache[cache_key] = load_prediction_model(model_path, network)
    if model_cache[cache_key] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found for {network}: {model_path}",
        )
    return model_cache[cache_key]


@app.get("/", response_class=HTMLResponse)
def web_app():
    """Serve the small browser UI used to upload images and choose a network."""
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
    button.secondary { background: #566b84; }
    button:disabled { background: #8291a3; cursor: wait; }
    .file-row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
    .file-row input { margin: 0; }
    .file-label { min-width: 86px; font-size: 14px; color: #334155; }
    .upload-actions { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
    pre { white-space: pre-wrap; background: #101820; color: #f4f7fb; padding: 14px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
  <main>
    <h1>skin_cancer_detector</h1>
    <section>
      <form id="form">
        <label>Skin lesion image</label>
        <div id="imageInputs"></div>
        <div id="zipInput"></div>
        <div class="upload-actions">
          <button id="addImage" class="secondary" type="button">Add Image</button>
          <button id="addZip" class="secondary" type="button">Upload Zip</button>
        </div>
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
    const imageInputs = document.getElementById("imageInputs");
    const zipInput = document.getElementById("zipInput");
    const addImage = document.getElementById("addImage");
    const addZip = document.getElementById("addZip");
    let latest = null;
    let imageCount = 0;

    function addImageInput() {
      imageCount += 1;
      const row = document.createElement("div");
      row.className = "file-row";
      row.innerHTML = `<span class="file-label">Image ${String(imageCount).padStart(3, "0")}</span>
        <input name="files" type="file" accept="image/*" />`;
      imageInputs.appendChild(row);
    }

    function addZipInput() {
      zipInput.innerHTML = `<div class="file-row"><span class="file-label">Zip file</span>
        <input name="files" type="file" accept=".zip" /></div>`;
    }

    addImage.addEventListener("click", addImageInput);
    addZip.addEventListener("click", addZipInput);
    addImageInput();

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
    """Return service state without running an expensive prediction."""
    return {
        "status": "ok",
        "networks": {
            network: str(get_model_path(network))
            for network in get_available_networks()
        },
        "model_urls_configured": {
            network: bool(os.getenv(config["url_env"]))
            for network, config in NETWORKS.items()
        },
        "loaded_models": sorted(model_cache),
        "vlm_provider": "ollama",
        "vlm_model": prediction_pipeline.vlm_agent.model,
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    network: Annotated[str, Form()] = "EfficientNetB3",
):
    """Run model inference for one uploaded image and attach report text."""
    try:
        network = normalize_network(network)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction_model = get_model(network)
    image_bytes = await file.read()
    result = prediction_pipeline.run(
        prediction_model,
        image_bytes,
        file.filename,
        file.content_type,
        network,
        str(get_model_path(network)),
    )
    return {"filename": file.filename, **result}


@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    network: Annotated[str, Form()] = "EfficientNetB3",
):
    """Run local model inference for multiple images or zipped image folders."""
    try:
        network = normalize_network(network)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    prediction_model = get_model(network)
    results = []
    for index, item in enumerate(image_items, start=1):
        result = prediction_pipeline.run(
            prediction_model,
            item["image_bytes"],
            item["filename"],
            item["content_type"],
            network,
            str(get_model_path(network)),
        )
        result["case_number"] = index
        result["case_id"] = f"Image {index:03d}"
        results.append({"filename": item["filename"], **result})

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
    """Convert an existing prediction result into a downloadable PDF report."""
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
