import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.tf_test_utils import disable_tensorflow_gpu_for_tests

disable_tensorflow_gpu_for_tests()

import numpy as np
import tensorflow as tf

from app.predict import (
    CLASS_NAMES,
    DISCLAIMER,
    extract_visual_features,
    get_available_networks,
    get_model_path,
    ensure_model_file,
    is_valid_keras_file,
    normalize_network,
    describe_uploaded_image,
    predict_image,
    predict_ensemble,
    preprocess_image,
)


class DummyModel:
    input_shape = (None, 16, 16, 3)

    def __call__(self, images, training=False):
        batch_size = int(images.shape[0])
        logits = np.zeros((batch_size, len(CLASS_NAMES)), dtype=np.float32)
        logits[:, 5] = 3.0
        return tf.constant(logits)


class PredictTests(unittest.TestCase):
    def test_preprocess_image_accepts_png_bytes(self):
        image = tf.zeros((8, 8, 3), dtype=tf.uint8)
        image_bytes = tf.io.encode_png(image).numpy()

        result = preprocess_image(image_bytes, img_size=16)

        self.assertEqual(tuple(result.shape), (1, 16, 16, 3))
        self.assertEqual(result.dtype, tf.float32)

    def test_predict_image_returns_contract_and_disclaimer(self):
        image = tf.zeros((8, 8, 3), dtype=tf.uint8)
        image_bytes = tf.io.encode_jpeg(image).numpy()

        result = predict_image(DummyModel(), image_bytes)

        self.assertEqual(result["prediction"], "MEL")
        self.assertIn("confidence", result)
        self.assertEqual(set(result["probabilities"].keys()), set(CLASS_NAMES))
        self.assertEqual(result["disclaimer"], DISCLAIMER)

    def test_describe_uploaded_image_returns_basic_image_details(self):
        image = tf.ones((8, 10, 3), dtype=tf.uint8) * 180
        image_bytes = tf.io.encode_jpeg(image).numpy()

        description = describe_uploaded_image(
            image_bytes,
            filename="lesion.jpg",
            content_type="image/jpeg",
        )

        self.assertEqual(description["filename"], "lesion.jpg")
        self.assertEqual(description["width"], 10)
        self.assertEqual(description["height"], 8)
        self.assertEqual(description["content_type"], "image/jpeg")
        self.assertIn("description", description)
        self.assertIn("visual_features", description)
        self.assertIn("summary", description["visual_features"])

    def test_default_model_path_points_to_keras_file(self):
        self.assertIn("best_model.keras", str(get_model_path()))
        self.assertIn("ResNet50/R_next", str(get_model_path("ResNet50")))
        self.assertIn("DenseNet121/D_next", str(get_model_path("DenseNet121")))
        self.assertEqual(str(get_model_path("Ensemble")), "Training_Sets/Ensemble")
        self.assertEqual(
            set(get_available_networks()),
            {"EfficientNetB3", "ResNet50", "DenseNet121", "Ensemble"},
        )
        self.assertEqual(normalize_network(None), "EfficientNetB3")

    def test_predict_ensemble_averages_probabilities(self):
        class FirstModel:
            input_shape = (None, 8, 8, 3)

            def __call__(self, images, training=False):
                return tf.constant([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        class SecondModel:
            input_shape = (None, 8, 8, 3)

            def __call__(self, images, training=False):
                return tf.constant([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        image = tf.zeros((8, 8, 3), dtype=tf.uint8)
        image_bytes = tf.io.encode_jpeg(image).numpy()
        probs = predict_ensemble({"a": FirstModel(), "b": SecondModel()}, image_bytes)

        self.assertEqual(len(probs), len(CLASS_NAMES))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_extract_visual_features_returns_explanation_fields(self):
        image = tf.ones((8, 8, 3), dtype=tf.float32) * 0.5

        features = extract_visual_features(image)

        self.assertIn("dominant_color", features)
        self.assertIn("edge_strength", features)
        self.assertIn("asymmetry", features)

    def test_ensure_model_file_downloads_when_url_is_configured(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self, size=-1):
                if getattr(self, "done", False):
                    return b""
                self.done = True
                return b"model-bytes"

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "best_model.keras"
            with patch.dict(
                "os.environ",
                {"SKIN_CANCER_EFFICIENTNET_MODEL_URL": "https://example.com/model.keras"},
            ):
                with patch("urllib.request.urlopen", return_value=FakeResponse()):
                    ensure_model_file(model_path, "EfficientNetB3")

            self.assertEqual(model_path.read_bytes(), b"model-bytes")

    def test_lfs_pointer_is_not_treated_as_valid_model(self):
        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "best_model.keras"
            model_path.write_text(
                "version https://git-lfs.github.com/spec/v1\n",
                encoding="utf-8",
            )

            self.assertFalse(is_valid_keras_file(model_path))


if __name__ == "__main__":
    unittest.main()
