import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from training.constants import NUM_CLASSES
from training.dataset import load_dataframe, make_balanced_dataset, make_dataset
from training.evaluate import evaluate_model, plot_training_curves
from training.model import (
    build_model,
    get_default_fine_tune_at,
    get_default_img_size,
    get_supported_networks,
)


"""Training entrypoint for HAM10000 skin lesion classification experiments."""

NETWORK_OUTPUT_DIRS = {
    "EfficientNetB3": "EfficientNetB3",
    "ResNet50": "ResNet50",
    "DenseNet121": "DenseNet121",
}
BASE_NETWORKS = tuple(NETWORK_OUTPUT_DIRS)
ENSEMBLE_NETWORK = "Ensemble"

DEFAULT_RUN_NAMES = {
    "EfficientNetB3": "T_next",
    "ResNet50": "R_next",
    "DenseNet121": "D_next",
}


def configure_tensorflow_runtime():
    """Enable GPU memory growth when TensorFlow can see a GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def parse_args():
    """Read CLI options and fill network-specific defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--network", choices=[*get_supported_networks(), ENSEMBLE_NETWORK], default="EfficientNetB3")
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--output_dir", default="Training_Sets")
    parser.add_argument("--Training_Sets_name")
    parser.add_argument("--fine_tune_at", type=int)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weights", default="imagenet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.network == ENSEMBLE_NETWORK:
        if args.weights.lower() == "none":
            args.weights = None
        return args

    if args.img_size is None:
        args.img_size = get_default_img_size(args.network)
    if args.fine_tune_at is None:
        args.fine_tune_at = get_default_fine_tune_at(args.network)
    if args.Training_Sets_name is None:
        args.Training_Sets_name = DEFAULT_RUN_NAMES[args.network]
    if args.weights.lower() == "none":
        args.weights = None

    return args


def prepare_output_dir(
    base_dir: str,
    network: str,
    training_set_name: str,
    overwrite: bool,
) -> Path:
    """Create the experiment output folder, protecting old runs by default."""
    output_dir = Path(base_dir) / NETWORK_OUTPUT_DIRS[network] / training_set_name

    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_config(args, output_dir: Path):
    """Save the training settings beside the model artifacts."""
    config = {
        "Training_Sets_name": args.Training_Sets_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.network,
        "dataset": "HAM10000",
        "num_classes": NUM_CLASSES,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "learning_rate": f"CosineDecay initial {args.learning_rate}",
        "loss": "SparseCategoricalFocalLoss",
        "focal_gamma": 2.0,
        "focal_alpha": 0.5,
        "fine_tune": True,
        "fine_tune_at": args.fine_tune_at,
        "weights": args.weights,
        "sampling": "Soft Balanced Sampling",
        "augmentation": [
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomTranslation",
            "RandomContrast",
        ],
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def main():
    """Train, evaluate, and save one selected network or all ensemble members."""
    configure_tensorflow_runtime()

    args = parse_args()
    if args.network == ENSEMBLE_NETWORK:
        train_ensemble(args)
        return

    train_single_network(args)


def train_single_network(args):
    """Train, evaluate, and save one network run."""
    output_dir = prepare_output_dir(
        args.output_dir,
        args.network,
        args.Training_Sets_name,
        args.overwrite,
    )

    df = load_dataframe(args.data_dir)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train_ds = make_balanced_dataset(
        train_df,
        args.img_size,
        args.batch_size,
    )
    val_ds = make_dataset(
        val_df["image_path"].values,
        val_df["label"].values,
        args.img_size,
        args.batch_size,
        training=False,
    )

    steps_per_epoch = max(1, len(train_df) // args.batch_size)

    model = build_model(
        img_size=args.img_size,
        num_classes=NUM_CLASSES,
        network=args.network,
        weights=args.weights,
        fine_tune=True,
        fine_tune_at=args.fine_tune_at,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    write_config(args, output_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    plot_training_curves(history, output_dir)
    evaluate_model(model, val_ds, output_dir)

    val_loss, val_acc = model.evaluate(val_ds)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Best model saved to: {output_dir / 'best_model.keras'}")


def train_ensemble(args):
    """Train all three ensemble member networks into their own network folders."""
    for network in BASE_NETWORKS:
        member_args = argparse.Namespace(**vars(args))
        member_args.network = network
        member_args.img_size = get_default_img_size(network)
        member_args.fine_tune_at = get_default_fine_tune_at(network)
        member_args.Training_Sets_name = (
            args.Training_Sets_name
            if args.Training_Sets_name
            else DEFAULT_RUN_NAMES[network]
        )
        print(f"\n=== Training ensemble member: {network} ===")
        train_single_network(member_args)


if __name__ == "__main__":
    main()
