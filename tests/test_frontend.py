import json
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import app.frontend as frontend


class FrontendTests(unittest.IsolatedAsyncioTestCase):
    def test_web_app_returns_html(self):
        html = frontend.web_app()

        self.assertIn("skin_cancer_detector", html)
        self.assertIn("EfficientNetB3", html)
        self.assertIn("Ensemble", html)

    def test_health_check_reports_proxy_mode(self):
        with patch.dict("os.environ", {"INFERENCE_API_URL": "https://example.hf.space"}):
            response = frontend.health_check()

        self.assertEqual(response["mode"], "frontend_proxy")
        self.assertTrue(response["inference_api_configured"])

    def test_get_inference_url_requires_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(HTTPException) as ctx:
                frontend.get_inference_url()

        self.assertEqual(ctx.exception.status_code, 503)

    async def test_predict_forwards_to_inference_api(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps({"prediction": "NV", "network": "ResNet50"}).encode("utf-8")

        upload = UploadFile(filename="lesion.jpg", file=BytesIO(b"image-bytes"))

        with patch.dict("os.environ", {"INFERENCE_API_URL": "https://example.hf.space"}):
            with patch("urllib.request.urlopen", return_value=FakeResponse()) as mock_urlopen:
                response = await frontend.predict(upload, network="ResNet50")

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.hf.space/predict")
        self.assertEqual(response["prediction"], "NV")
        self.assertEqual(response["network"], "ResNet50")

    def test_build_multipart_body_includes_network_and_file(self):
        body, content_type = frontend.build_multipart_body(
            image_bytes=b"abc",
            filename="lesion.jpg",
            file_content_type="image/jpeg",
            network="DenseNet121",
        )

        self.assertIn("multipart/form-data", content_type)
        self.assertIn(b'name="network"', body)
        self.assertIn(b"DenseNet121", body)
        self.assertIn(b'filename="lesion.jpg"', body)
        self.assertIn(b"abc", body)


if __name__ == "__main__":
    unittest.main()
