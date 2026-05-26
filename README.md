# skin_cancer_detector

Educational skin lesion image classification web app using TensorFlow/Keras CNN
models, a lightweight multi-agent pipeline, and local Ollama/LLaVA visual
description.

Test site:

```text
https://skin-cancer-detector.com
```

This project is for learning and research only. It is not a medical diagnostic
system.

## Workflow

```text
User uploads a skin lesion image
  -> selects a prediction network
  -> CNN model predicts one HAM10000 class
  -> local image feature agent extracts visual cues
  -> local LLaVA VLM adds a non-diagnostic image description when available
  -> safety agent adds the medical disclaimer
  -> report agent generates text + PDF report
```

The web app also supports batch input. Users can click `Add Image` to add image
files one by one, remove selected inputs, or click `Upload Zip` to submit a
zipped image folder. Batch reports are generated in order as `Image 001`,
`Image 002`, and so on.

Supported networks:

```text
EfficientNetB3
ResNet50
DenseNet121
Ensemble
```

`Ensemble` combines EfficientNetB3, ResNet50, and DenseNet121 by averaging their
prediction probabilities.

Predicted classes:

```text
AKIEC, BCC, BKL, DF, NV, MEL, VASC
```

## Main Technologies

```text
TensorFlow / Keras   CNN training and inference
FastAPI / Uvicorn    Web app and API
Ollama + LLaVA       Local open-source VLM image description
Python agents        CNN, image feature, VLM, safety, and report agents
PDF generation       Downloadable educational report
```

## Project Layout

```text
app/          Web API, prediction pipeline, agents, reports
training/     Model training, dataset loading, evaluation
tests/        Unit tests
dataset/      Local HAM10000 data
Training_Sets/ Trained model results grouped by network
```

Model results are organized as:

```text
Training_Sets/EfficientNetB3/
Training_Sets/ResNet50/
Training_Sets/DenseNet121/
```

## Run Locally

```bash
source venv/bin/activate
ollama serve
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

If port 8000 is busy:

```bash
uvicorn app.api:app --reload --port 8001
```

## VLM Availability

Local deployment can use Ollama + LLaVA for VLM image-description testing when
Ollama is running on the same machine. In the cloud deployment, VLM is currently
disabled because the project does not call a hosted VLM API. The online version
still supports CNN prediction, local image feature extraction, safety text, and
PDF report generation.

For the lightweight Render frontend, run:

```bash
INFERENCE_API_URL=https://your-huggingface-space.hf.space uvicorn app.frontend:app --reload
```

## Train A Model

```bash
python -m training.train \
  --data_dir dataset \
  --network ResNet50 \
  --Training_Sets_name R_next \
  --epochs 20 \
  --overwrite
```

Train all three ensemble member networks:

```bash
python -m training.train \
  --data_dir dataset \
  --network Ensemble \
  --epochs 20 \
  --overwrite
```

Training output is saved to:

```text
Training_Sets/<network>/<run_name>/
```

## Upload Trained Models

The trained `.keras` model files are large, so they are not stored directly in
the GitHub repository. After training, upload the active models to Hugging Face
with:

```bash
pip install huggingface_hub
hf auth login
python upload_models.py
```

The script uploads:

```text
Training_Sets/EfficientNetB3/T_next/best_model.keras -> EfficientNetB3/best_model.keras
Training_Sets/ResNet50/R_next/best_model.keras       -> ResNet50/best_model.keras
Training_Sets/DenseNet121/D_next/best_model.keras    -> DenseNet121/best_model.keras
```

The deployed app can download these model files from Hugging Face when local
model files are missing.

## Deployment Split

The cloud version uses Hugging Face Space for model inference and Render for the
public web frontend:

```text
User browser
  -> Render frontend (app.frontend)
  -> Hugging Face Space inference API (app.api)
  -> Hugging Face model files (.keras)
  -> Hugging Face Space returns prediction/report JSON
  -> Render displays result and generates downloadable PDF
```

Render stays lightweight and does not load TensorFlow models. Hugging Face Space
runs TensorFlow inference, downloads the hosted model files when needed, and
returns the prediction result to Render.

Set `INFERENCE_API_URL` on Render to the Hugging Face Space URL.

Render should run only the lightweight frontend:

```text
Build Command: pip install -r requirements-frontend.txt
Start Command: uvicorn app.frontend:app --host 0.0.0.0 --port $PORT
```

After deployment, open `/health`. A correct Render frontend returns:

```json
{"status":"ok","mode":"frontend_proxy","inference_api_configured":true}
```

If `/health` returns model paths or `loaded_models`, Render is still running
`app.api` instead of `app.frontend`.

View `Training_Report.docx` for more information.
