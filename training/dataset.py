from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np

"""HAM10000 metadata loading and TensorFlow dataset helpers."""

# Convert HAM10000 diagnosis strings to integer class IDs.
LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "nv": 4,
    "mel": 5,
    "vasc": 6,
}

def find_image_path(base_dir: Path, image_id: str):
    """Find a HAM10000 image in either image-part directory."""

    candidates = [
        base_dir / "HAM10000_images_part_1" / f"{image_id}.jpg",
        base_dir / "HAM10000_images_part_2" / f"{image_id}.jpg",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Image not found: {image_id}")

def make_balanced_dataset(df, img_size, batch_size):
    """Create a repeated balanced dataset for training on imbalanced classes."""
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        return image, label

    class_datasets = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]

        ds = tf.data.Dataset.from_tensor_slices(
            (
                class_df["image_path"].values,
                class_df["label"].values,
            )
        )

        ds = ds.shuffle(len(class_df), reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

        class_datasets.append(ds)

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[
            0.13,  # AKIEC
            0.13,  # BCC
            0.14,  # BKL
            0.10,  # DF
            0.25,  # NV
            0.15,  # MEL
            0.10,  # VASC
        ]
    )

    balanced_ds = balanced_ds.batch(batch_size)
    balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)

    return balanced_ds

def load_dataframe(data_dir: str):
    """Load metadata and attach integer labels plus image file paths."""

    base_dir = Path(data_dir)

    csv_path = base_dir / "HAM10000_metadata.csv"

    df = pd.read_csv(csv_path)

    df["label"] = df["dx"].map(LABEL_MAP)

    df["image_path"] = df["image_id"].apply(
        lambda x: find_image_path(base_dir, x)
    )

    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(np.int64)

    return df

def make_dataset(paths, labels, img_size, batch_size, training=False):
    """Create a standard batched TensorFlow dataset from image paths."""

    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.resize(image, [img_size, img_size])

        image = tf.cast(image, tf.float32)

        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2000)

    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
