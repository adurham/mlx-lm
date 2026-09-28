"""EXL3 trellis kernels vendored from PonyExl3 (Apache-2.0) for mlx-lm.

Import surface: ``EXL3SwitchGLU`` (MoE expert switch GLU) and ``EXL3Linear``
(dense projections loaded from EXL3 tensor groups).

See LICENSE, NOTICE and README.md in this directory for provenance and the
list of local changes.
"""

from .exl3_linear import EXL3Linear
from .exl3_moe import EXL3SwitchGLU

__all__ = ["EXL3Linear", "EXL3SwitchGLU"]
