import os


def disable_tensorflow_gpu_for_tests():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
