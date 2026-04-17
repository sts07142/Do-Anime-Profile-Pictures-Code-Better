"""Auto-detect best available device for inference."""

import torch


def get_device() -> str:
    """Return the best available device string for ultralytics/torch.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
