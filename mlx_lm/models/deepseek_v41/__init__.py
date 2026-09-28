"""DeepSeek-V4.1-Flash (text stack) for MLX.

Model code vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0, LICENSE
here) plus local changes listed in README.md. ``exl3_build`` builds the model
from an EXL3 checkpoint.
"""

from .config import ModelArgs
from .model import Model

__all__ = ["Model", "ModelArgs"]
