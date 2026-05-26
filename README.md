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

## Quick API Test

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@lesion.jpg" -F "network=ResNet50" http://127.0.0.1:8000/predict
curl -F "file=@lesion.jpg" -F "network=Ensemble" http://127.0.0.1:8000/predict
```

## GitHub / Cloud Deploy

Do not commit `venv/`, `dataset/`, or `Training_Sets/`. Upload trained model
files as deployment artifacts or set the model path environment variables on the
server. The app can start with:

```bash
uvicorn app.api:app --host 0.0.0.0 --port $PORT
```

For VLM descriptions, the deployed service also needs access to Ollama/LLaVA via
`OLLAMA_URL` and `OLLAMA_VLM_MODEL`.

## Verify

```bash
./venv/bin/python -m py_compile app/*.py training/*.py tests/*.py
./venv/bin/python -m unittest discover
```
