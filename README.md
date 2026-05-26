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

View `Training_Report.docx` for more information.
