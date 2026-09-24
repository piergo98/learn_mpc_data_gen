"""Solver-like adapter for Scholtes MPCC relaxations solved with IPOPT."""

import casadi as ca


class ScholtesHomotopySolver:
    """Apply an IPOPT solver repeatedly along a decreasing relaxation path.

    The wrapped NLP has one extra scalar parameter (the Scholtes relaxation
    ``tau``) after the parameters exposed to callers.  Keeping that parameter
    private makes this object compatible with the ``nlpsol`` interface used by
    the bilevel sampling routines.
    """

    def __init__(self, solver, parameter_size, relaxation_values):
        values = tuple(float(value) for value in relaxation_values)
        if not values:
            raise ValueError("scholtes_relaxations must contain at least one value.")
        if any(value <= 0.0 for value in values):
            raise ValueError("Scholtes relaxation values must be positive.")
        if any(next_value >= value for value, next_value in zip(values, values[1:])):
            raise ValueError("scholtes_relaxations must be strictly decreasing.")

        self._solver = solver
        self._parameter_size = int(parameter_size)
        self.relaxation_values = values
        self._stats = {}

    def __call__(self, *args, **kwargs):
        if args:
            raise TypeError(
                "ScholtesHomotopySolver accepts the same keyword arguments used "
                "by this project's bilevel samplers; positional arguments are not supported."
            )

        call_args = dict(kwargs)
        user_parameters = call_args.pop("p", ca.DM.zeros(self._parameter_size, 1))
        user_parameters = ca.vec(ca.DM(user_parameters))
        if user_parameters.numel() != self._parameter_size:
            raise ValueError(
                f"Expected {self._parameter_size} NLP parameters, got "
                f"{user_parameters.numel()}."
            )

        solution = None
        completed_relaxations = []
        for relaxation in self.relaxation_values:
            stage_args = dict(call_args)
            stage_args["p"] = ca.vertcat(user_parameters, relaxation)
            if solution is not None:
                stage_args["x0"] = solution["x"]

            solution = self._solver(**stage_args)
            stage_stats = dict(self._solver.stats())
            completed_relaxations.append(relaxation)
            if not stage_stats.get("success", False):
                break

        self._stats = dict(self._solver.stats())
        self._stats["scholtes_relaxations"] = completed_relaxations
        self._stats["scholtes_final_relaxation"] = completed_relaxations[-1]
        self._stats["scholtes_homotopy_complete"] = (
            len(completed_relaxations) == len(self.relaxation_values)
            and bool(self._stats.get("success", False))
        )
        return solution

    def stats(self):
        """Return the final IPOPT statistics plus homotopy diagnostics."""
        return dict(self._stats)
