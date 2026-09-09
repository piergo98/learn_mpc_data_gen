"""Default linear-MPC problem data used by the sampling experiments.

These constants are the same double-integrator-like system used in the main
``ultro`` repository (``src/linear_training_strategy_comparison.py``). They are
duplicated here so that this ``sampling`` folder stays completely independent
from the rest of the repository.
"""

from __future__ import annotations

import numpy as np

DEFAULT_A = np.array([[1.0, 0.1], [0.981, 1.0]], dtype=float)
DEFAULT_B = np.array([[0.0], [0.1]], dtype=float)
DEFAULT_Q = np.diag([30.0, 1.0])
DEFAULT_R = np.diag([1.0])
DEFAULT_X_BOUNDS = np.array([2.5, 3.5], dtype=float)
DEFAULT_U_BOUNDS = np.array([2.0], dtype=float)

__all__ = [
    "DEFAULT_A",
    "DEFAULT_B",
    "DEFAULT_Q",
    "DEFAULT_R",
    "DEFAULT_X_BOUNDS",
    "DEFAULT_U_BOUNDS",
]
