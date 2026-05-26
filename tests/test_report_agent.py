import unittest

from app.report_agent import generate_batch_markdown_report, generate_markdown_report, markdown_to_pdf_bytes


class ReportAgentTests(unittest.TestCase):
    def test_generate_markdown_report_includes_prediction_and_disclaimer(self):
        report = generate_markdown_report(
            {
                "prediction": "MEL",
                "confidence": 0.72,
                "probabilities": {"MEL": 0.72, "NV": 0.20, "BCC": 0.08},
                "image": {
                    "filename": "lesion.jpg",
                    "content_type": "image/jpeg",
                    "size_bytes": 1234,
                    "width": 10,
                    "height": 8,
                    "description": "The uploaded file is lesion.jpg, 10x8 pixels.",
                    "visual_features": {
                        "summary": (
                            "reddish or pink-toned color, high color variation, "
                            "moderate border/edge change, and moderate asymmetry."
                        )
                    },
                },
                "disclaimer": "Not a medical diagnosis.",
                "vlm_analysis": {
                    "available": True,
                    "provider": "openai",
                    "model": "test-vlm",
                    "description": "The lesion appears dark with irregular texture.",
                },
            }
        )

        self.assertIn("# Skin Lesion AI Report", report)
        self.assertIn("Predicted class: Melanoma", report)
        self.assertIn("Model class code: MEL", report)
        self.assertIn("lesion.jpg", report)
        self.assertIn("10x8", report)
        self.assertIn("Image Feature Extraction", report)
        self.assertIn("convolutional neural network", report)
        self.assertIn("reddish or pink-toned", report)
        self.assertIn("VLM Image Description", report)
        self.assertIn("irregular texture", report)
        self.assertIn("Not a medical diagnosis.", report)

    def test_markdown_to_pdf_bytes_returns_pdf_document(self):
        pdf = markdown_to_pdf_bytes("# Report\n\nPrediction: MEL")

        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"%%EOF", pdf)

    def test_report_summarizes_disabled_vlm_cleanly(self):
        report = generate_markdown_report(
            {
                "prediction": "NV",
                "confidence": 0.8,
                "vlm_analysis": {
                    "available": False,
                    "reason": "VLM is disabled for this deployment.",
                },
            }
        )

        self.assertIn("VLM image description is disabled", report)
        self.assertNotIn("Reason:", report)

    def test_generate_batch_markdown_report_numbers_results(self):
        report = generate_batch_markdown_report(
            [
                {
                    "case_id": "Image 001",
                    "filename": "a.jpg",
                    "prediction": "NV",
                    "confidence": 0.7,
                    "probabilities": {"NV": 0.7, "MEL": 0.3},
                },
                {
                    "case_id": "Image 002",
                    "filename": "b.jpg",
                    "prediction": "MEL",
                    "confidence": 0.8,
                    "probabilities": {"MEL": 0.8, "NV": 0.2},
                },
            ]
        )

        self.assertIn("# Skin Lesion AI Batch Report", report)
        self.assertIn("## Image 001 - a.jpg", report)
        self.assertIn("## Image 002 - b.jpg", report)
        self.assertIn("Total images: 2", report)


if __name__ == "__main__":
    unittest.main()
