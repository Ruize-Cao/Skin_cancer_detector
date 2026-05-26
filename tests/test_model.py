import unittest

from tests.tf_test_utils import disable_tensorflow_gpu_for_tests

disable_tensorflow_gpu_for_tests()

from training.model import (
    DEFAULT_IMG_SIZE,
    build_model,
    get_default_img_size,
    get_supported_networks,
)


class ModelTests(unittest.TestCase):
    def test_default_image_size_matches_saved_model_family(self):
        self.assertEqual(DEFAULT_IMG_SIZE, 300)
        self.assertEqual(get_default_img_size("ResNet50"), 224)
        self.assertEqual(get_default_img_size("DenseNet121"), 224)

    def test_supported_networks_include_training_choices(self):
        self.assertEqual(
            set(get_supported_networks()),
            {"EfficientNetB3", "ResNet50", "DenseNet121"},
        )

    def test_build_model_creates_classifier(self):
        for network in get_supported_networks():
            with self.subTest(network=network):
                model = build_model(
                    img_size=64,
                    num_classes=7,
                    network=network,
                    weights=None,
                    fine_tune=False,
                    epochs=1,
                    steps_per_epoch=1,
                )

                self.assertEqual(model.input_shape, (None, 64, 64, 3))
                self.assertEqual(model.output_shape, (None, 7))


if __name__ == "__main__":
    unittest.main()
