import json
import unittest
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

from fastapi import HTTPException, UploadFile

import app.frontend as frontend


class FrontendTests(unittest.IsolatedAsyncioTestCase):
    def test_web_app_returns_html(self):
        html = frontend.web_app()

        self.assertIn("skin_cancer_detector", html)
        self.assertIn("EfficientNetB3", html)
        self.assertIn("Ensemble", html)
        self.assertIn("multiple", html)
        self.assertIn("/predict-batch", html)

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

    async def test_predict_batch_returns_ordered_results(self):
        responses = [
            {"prediction": "NV", "confidence": 0.7},
            {"prediction": "MEL", "confidence": 0.8},
        ]

        with patch.dict("os.environ", {"INFERENCE_API_URL": "https://example.hf.space"}):
            with patch("app.frontend.forward_image_to_inference", side_effect=responses):
                result = await frontend.predict_batch(
                    files=[
                        UploadFile(filename="a.jpg", file=BytesIO(b"a")),
                        UploadFile(filename="b.png", file=BytesIO(b"b")),
                    ],
                    network="EfficientNetB3",
                )

        self.assertTrue(result["batch"])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["results"][0]["case_id"], "Image 001")
        self.assertEqual(result["results"][1]["case_id"], "Image 002")
        self.assertIn("Skin Lesion AI Batch Report", result["text_report"])

    async def test_predict_batch_returns_single_result_for_one_image(self):
        with patch.dict("os.environ", {"INFERENCE_API_URL": "https://example.hf.space"}):
            with patch("app.frontend.forward_image_to_inference", return_value={"prediction": "NV"}):
                result = await frontend.predict_batch(
                    files=[UploadFile(filename="single.jpg", file=BytesIO(b"a"))],
                    network="ResNet50",
                )

        self.assertEqual(result["prediction"], "NV")
        self.assertEqual(result["case_id"], "Image 001")
        self.assertNotIn("batch", result)

    async def test_extract_upload_images_reads_zip_in_filename_order(self):
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as archive:
            archive.writestr("b.png", b"b")
            archive.writestr("folder/a.jpg", b"a")
            archive.writestr("notes.txt", b"skip")

        upload = UploadFile(filename="images.zip", file=BytesIO(zip_buffer.getvalue()))
        items = await frontend.extract_upload_images(upload)

        self.assertEqual([item["filename"] for item in items], ["b.png", "folder/a.jpg"])
        self.assertEqual(items[0]["content_type"], "image/png")
        self.assertEqual(items[1]["content_type"], "image/jpeg")

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

    def test_report_pdf_handles_batch_result(self):
        response = frontend.report_pdf(
            {
                "batch": True,
                "results": [
                    {
                        "case_id": "Image 001",
                        "filename": "a.jpg",
                        "prediction": "NV",
                        "confidence": 0.7,
                    }
                ],
            }
        )

        self.assertEqual(response.media_type, "application/pdf")
        self.assertIn("skin-lesion-ai-batch-report.pdf", response.headers["content-disposition"])


if __name__ == "__main__":
    unittest.main()
