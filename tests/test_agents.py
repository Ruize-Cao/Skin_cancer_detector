import unittest
from unittest.mock import patch

from app.agents import SafetyAgent, VLMImageDescriptionAgent
from app.report_agent import DISCLAIMER


class AgentTests(unittest.TestCase):
    def test_vlm_agent_falls_back_when_ollama_is_unavailable(self):
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            result = VLMImageDescriptionAgent().run(b"fake-image", "image/jpeg")

        self.assertFalse(result["available"])
        self.assertEqual(result["provider"], "ollama")
        self.assertIn("offline", result["reason"])

    def test_vlm_agent_returns_ollama_description(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"response": "dark border and mixed color"}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            result = VLMImageDescriptionAgent(model="llava").run(b"fake-image", "image/jpeg")

        self.assertTrue(result["available"])
        self.assertEqual(result["provider"], "ollama")
        self.assertIn("mixed color", result["description"])

    def test_safety_agent_adds_disclaimer(self):
        result = SafetyAgent().run({})

        self.assertEqual(result["disclaimer"], DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
