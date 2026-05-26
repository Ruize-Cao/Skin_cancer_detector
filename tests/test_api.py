import unittest
from io import BytesIO
from unittest.mock import patch

from tests.tf_test_utils import disable_tensorflow_gpu_for_tests

disable_tensorflow_gpu_for_tests()

from fastapi import HTTPException, UploadFile
import tensorflow as tf

import app.api as api


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def test_web_app_returns_html(self):
        response = api.web_app()

        self.assertIn("skin_cancer_detector", response)
        self.assertIn("Skin lesion image", response)
        self.assertIn("EfficientNetB3", response)
        self.assertIn("ResNet50", response)
        self.assertIn("DenseNet121", response)
        self.assertIn("Ensemble", response)
        self.assertNotIn("Train Model", response)

    def test_health_check_returns_core_state(self):
        response = api.health_check()

        self.assertEqual(response["status"], "ok")
        self.assertIn("networks", response)
        self.assertIn("loaded_models", response)
        self.assertIn("ResNet50", response["networks"])
        self.assertIn("Ensemble", response["networks"])
        self.assertIn("disclaimer", response)

    async def test_predict_returns_503_when_model_is_missing(self):
        previous_cache = dict(api.model_cache)
        api.model_cache.clear()

        try:
            with patch("app.api.get_model_path") as mock_path, patch(
                "app.api.load_prediction_model"
            ) as mock_loader:
                mock_path.return_value = "missing.keras"
                mock_loader.return_value = None

                with self.assertRaises(HTTPException) as ctx:
                    await api.predict(UploadFile(filename="lesion.jpg", file=None))
        finally:
            api.model_cache.clear()
            api.model_cache.update(previous_cache)

        self.assertEqual(ctx.exception.status_code, 503)

    async def test_predict_returns_prediction_and_report(self):
        class DummyModel:
            input_shape = (None, 16, 16, 3)

            def __call__(self, images, training=False):
                return tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]])

        image_bytes = tf.io.encode_jpeg(tf.zeros((8, 8, 3), dtype=tf.uint8)).numpy()
        upload = UploadFile(filename="lesion.jpg", file=BytesIO(image_bytes))

        with patch("app.api.get_model") as mock_model, patch(
            "app.api.get_model_path"
        ) as mock_path, patch(
            "app.api.prediction_pipeline.vlm_agent.run"
        ) as mock_vlm:
            mock_model.return_value = DummyModel()
            mock_path.return_value = "dummy.keras"
            mock_vlm.return_value = {
                "available": False,
                "provider": "ollama",
                "reason": "mocked in test",
            }
            response = await api.predict(upload, network="ResNet50")

        self.assertEqual(response["prediction"], "MEL")
        self.assertEqual(response["network"], "ResNet50")
        self.assertIn("image", response)
        self.assertEqual(response["image"]["width"], 8)
        self.assertIn("visual_features", response["image"])
        self.assertIn("vlm_analysis", response)
        self.assertFalse(response["vlm_analysis"]["available"])
        self.assertIn("text_report", response)
        self.assertIn("Predicted class: Melanoma", response["text_report"])
        self.assertIn("Image Feature Extraction", response["text_report"])
        self.assertIn("VLM Image Description", response["text_report"])

    def test_report_endpoints_generate_markdown_and_pdf(self):
        result = {
            "prediction": "MEL",
            "confidence": 0.7,
            "probabilities": {"MEL": 0.7, "NV": 0.3},
        }

        pdf = api.report_pdf(result)

        self.assertEqual(pdf.media_type, "application/pdf")
        self.assertTrue(pdf.body.startswith(b"%PDF-1.4"))


if __name__ == "__main__":
    unittest.main()
