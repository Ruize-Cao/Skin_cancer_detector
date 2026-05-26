import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

from training.focal_loss import SparseCategoricalFocalLoss


"""Model builders for the selectable CNN backbones used in training."""

# Each network keeps its own preprocessing function and default image size.
NETWORKS = {
    "EfficientNetB3": {
        "builder": tf.keras.applications.EfficientNetB3,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "img_size": 300,
        "fine_tune_at": 300,
    },
    "ResNet50": {
        "builder": tf.keras.applications.ResNet50,
        "preprocess": tf.keras.applications.resnet.preprocess_input,
        "img_size": 224,
        "fine_tune_at": 140,
    },
    "DenseNet121": {
        "builder": tf.keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input,
        "img_size": 224,
        "fine_tune_at": 300,
    },
}

DEFAULT_NETWORK = "EfficientNetB3"
DEFAULT_IMG_SIZE = NETWORKS[DEFAULT_NETWORK]["img_size"]
DEFAULT_FINE_TUNE_AT = NETWORKS[DEFAULT_NETWORK]["fine_tune_at"]


def get_supported_networks() -> list[str]:
    """Return network names accepted by the training CLI."""
    return list(NETWORKS)


def get_default_img_size(network: str) -> int:
    """Return the default input image size for a network."""
    return NETWORKS[network]["img_size"]


def get_default_fine_tune_at(network: str) -> int:
    """Return the layer index where fine-tuning starts by default."""
    return NETWORKS[network]["fine_tune_at"]


def build_model(
    img_size: int = DEFAULT_IMG_SIZE,
    num_classes: int = 7,
    network: str = DEFAULT_NETWORK,
    weights: str | None = "imagenet",
    fine_tune: bool = True,
    fine_tune_at: int | None = None,
    learning_rate: float = 3e-5,
    epochs: int = 20,
    steps_per_epoch: int = 1,
):
    """Build and compile a classifier using the selected pretrained backbone."""
    if network not in NETWORKS:
        raise ValueError(f"Unsupported network: {network}")

    config = NETWORKS[network]
    if fine_tune_at is None:
        fine_tune_at = config["fine_tune_at"]

    base_model = config["builder"](
        include_top=False,
        weights=weights,
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = fine_tune
    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    # Augmentation runs inside the model so it is active during training only.
    x = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.20),
            layers.RandomTranslation(0.10, 0.10),
            layers.RandomContrast(0.20),
        ],
        name="augmentation",
    )(inputs)
    x = config["preprocess"](x)
    x = base_model(x, training=fine_tune)
    # Small classification head shared by all backbone choices.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(
            learning_rate=CosineDecay(
                initial_learning_rate=learning_rate,
                decay_steps=max(1, epochs * steps_per_epoch),
                alpha=0.1,
            )
        ),
        loss=SparseCategoricalFocalLoss(gamma=2.0, alpha=0.5, from_logits=True),
        metrics=["accuracy"],
    )
    return model
