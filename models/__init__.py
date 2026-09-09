"""Dynamical system models used by the sampling experiments.

Each module defines a self-contained CasADi model that exposes both the MPC
problem and the feasible-initial-state samplers compared in ``sampling/src``.
"""

__all__ = [
    "CartPole",
    "DiscontinuousControl",
    "LinearSystem",
    "PointMass",
]
