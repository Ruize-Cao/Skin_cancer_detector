from pathlib import Path

from huggingface_hub import create_repo, upload_file


"""Upload the active Keras model files to the Hugging Face model repository."""

REPO_ID = "Ruizecao/skin-cancer-ai-models"
MODEL_FILES = [
    (
        Path("Training_Sets/EfficientNetB3/T_next/best_model.keras"),
        "EfficientNetB3/best_model.keras",
    ),
    (
        Path("Training_Sets/ResNet50/R_next/best_model.keras"),
        "ResNet50/best_model.keras",
    ),
    (
        Path("Training_Sets/DenseNet121/D_next/best_model.keras"),
        "DenseNet121/best_model.keras",
    ),
]


def main() -> None:
    """Create the Hugging Face repo if needed and upload the active models."""
    missing = [str(local_path) for local_path, _ in MODEL_FILES if not local_path.exists()]
    if missing:
        raise SystemExit("Missing model files:\n" + "\n".join(missing))

    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    for local_path, repo_path in MODEL_FILES:
        print(f"Uploading {local_path} -> {repo_path}")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )


if __name__ == "__main__":
    main()
