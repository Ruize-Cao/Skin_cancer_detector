import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.train import DEFAULT_RUN_NAMES, prepare_output_dir, write_config


class TrainPathTests(unittest.TestCase):
    def test_prepare_output_dir_groups_runs_by_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = prepare_output_dir(
                base_dir=tmp,
                network="ResNet50",
                training_set_name="R_test",
                overwrite=False,
            )

            self.assertEqual(output_dir, Path(tmp) / "ResNet50" / "R_test")
            self.assertTrue(output_dir.exists())

    def test_write_config_records_selected_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                Training_Sets_name="D_test",
                network="DenseNet121",
                img_size=224,
                batch_size=16,
                epochs=1,
                learning_rate=3e-5,
                fine_tune_at=300,
                weights=None,
            )

            write_config(args, Path(tmp))

            self.assertIn('"model": "DenseNet121"', (Path(tmp) / "config.json").read_text())

    def test_ensemble_default_run_names_cover_all_members(self):
        self.assertEqual(
            set(DEFAULT_RUN_NAMES),
            {"EfficientNetB3", "ResNet50", "DenseNet121"},
        )


if __name__ == "__main__":
    unittest.main()
