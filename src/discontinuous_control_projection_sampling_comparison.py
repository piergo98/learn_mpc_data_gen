"""Compare projection- and rollout-based sampling for the discontinuous-control MPC.

The experiment compares four strategies for generating feasible initial states:

1. ``uniform``            -- uniform candidates in the state box kept only when
   the original MPC is feasible for them.
2. ``bilevel_projection`` -- candidates from the bilevel active-set sampler,
   each one afterwards mapped through the nonlinear feasible-set projection.
3. ``slacked_projection`` -- candidates from the slacked extreme-state sampler
   (``seek_x0`` with randomly spiked residual weights), each one afterwards
   mapped through the same projection.
4. ``bilevel_rollout``    -- bilevel seeds propagated with closed-loop MPC
   rollouts, so every visited state becomes a sample.

For every method the script reports accepted states, wall-clock time split into
generation and projection/rollout phases, attempts, how often the projection
actually moved a candidate, and a post-hoc feasibility check.

Rejections are reported by cause rather than as one rate. The bilevel sampler
cannot return an infeasible state by construction, so it can only lose a
candidate to a failed solve or to its reduced-Hessian screen; a rejection there
means the solution is not a strict local minimizer. The same second-order
sufficient condition is evaluated afterwards for every method by
:func:`check_sosc`, so the samplers that do not screen for optimality
themselves are measured on the same footing.

Run from the repository root, for example::

    python3 src/discontinuous_control_projection_sampling_comparison.py --num-samples 100

Results are written to ``results/discontinuous_control_projection_sampling`` by
default. The bilevel methods require the CasADi ``ccopt`` plugin used by the
model.

Every state-plane panel is drawn on top of the exact MPC feasible set, which
this model admits in closed form (see :func:`feasible_set_profile`).

Notes
-----
* First controls are recomputed once after all timings, so the reported
  ``u_0^*`` values are consistent across methods and never bias wall times.
* Closed-loop rollouts use a local MPC first-control helper rather than
  ``DiscontinuousControl.solve_MPC``, which currently returns the first entry
  of the decision vector (a state component) instead of the first input.
* Every MPC solve starts from a guess with nonzero inputs. The model's stored
  guess sets all inputs to zero, which is the one point where the terminal
  constraint's Jacobian loses rank, and IPOPT then reports feasible states as
  ``Infeasible_Problem_Detected`` (see :func:`mpc_warm_starts`).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.discontinuous_control import DiscontinuousControl


LABELS = {
    "uniform": "Uniform + MPC check",
    "bilevel_projection": "Bilevel active set + projection",
    "slacked_projection": "Slacked extreme states + projection",
    "bilevel_rollout": "Bilevel + closed-loop rollouts",
}
COLORS = {
    "uniform": "tab:green",
    "bilevel_projection": "tab:red",
    "slacked_projection": "tab:blue",
    "bilevel_rollout": "tab:orange",
}
# A projection displacement below this norm means the candidate was already on
# the feasible set, so the projection only re-solved the MPC.
PROJECTION_MOVE_TOL = 1e-8
# Resolution of the analytic feasible-set outline. The boundary is piecewise
# smooth, so this is a plotting choice rather than an accuracy limit.
FEASIBLE_SET_RESOLUTION = 1201
# Extra randomized MPC initial guesses tried before a state is called
# infeasible. See :func:`mpc_warm_starts` for why the first guess matters: it
# is rank-safe and in practice always suffices, so the restarts are only a
# safety margin and are paid almost exclusively on infeasible candidates.
MPC_RESTARTS = 2
# A closed-loop rollout stops once it reaches the origin: with the terminal
# constraint x_N = 0 the state stays there, so further steps only duplicate it.
ROLLOUT_ORIGIN_TOL = 1e-9
# Bound on the seed batches drawn to reach the requested sample count. Each
# batch re-enters the sampler, which rebuilds its solver, so the top-up should
# converge in a couple of rounds rather than creep up on the target.
ROLLOUT_MAX_BATCHES = 8
# Safety factor on the seed count of a top-up batch. The marginal yield keeps
# falling as states accumulate, so asking for exactly the projected number of
# seeds reliably lands just short.
ROLLOUT_SEED_MARGIN = 1.35


@dataclass
class SamplingResult:
    method: str
    samples: np.ndarray
    elapsed_s: float
    attempts: int
    generation_s: float = 0.0
    post_processing_s: float = 0.0
    candidates: Optional[np.ndarray] = None
    candidate_feasible: Optional[np.ndarray] = None
    projection_attempts: int = 0
    projection_moved: int = 0
    projection_failures: int = 0
    projection_shifts: np.ndarray = field(default_factory=lambda: np.empty(0))
    rollout_seeds: int = 0
    rollout_steps: int = 0
    rollout_failures: int = 0
    rollout_left_box: int = 0
    rollout_converged: int = 0
    rollout_duplicates: int = 0
    rollout_batches: int = 0
    trajectories: List[np.ndarray] = field(default_factory=list)
    controls: np.ndarray = field(default_factory=lambda: np.empty(0))
    verified_feasible: int = 0
    analytic_feasible: int = 0
    # Rejections are split by cause: a candidate outside the feasible set, a
    # solver that failed on a feasible one, and a solution that solved but is
    # not a strict local minimizer (SOSC).
    infeasible_rejections: int = 0
    solver_failures: int = 0
    optimality_rejections: int = 0
    locally_optimal: int = 0
    sosc_degenerate: int = 0
    mpc_restarts: int = 0
    # NLP solves consumed by the timed phases, and by the untimed verification.
    solver_calls: int = 0
    verification_solver_calls: int = 0

    @property
    def accepted(self) -> int:
        return int(self.samples.shape[0])

    @property
    def rejected(self) -> int:
        return (
            self.infeasible_rejections
            + self.solver_failures
            + self.optimality_rejections
            + self.projection_failures
        )

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempts if self.attempts else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Accepted samples per attempt -- the complement a reader expects,
        not necessarily ``1 - rejection_rate`` since ``rejected`` is a sum of
        causes tracked for diagnosis rather than attempts - accepted."""
        return self.accepted / self.attempts if self.attempts else float("nan")

    @property
    def solver_calls_per_attempt(self) -> float:
        return self.solver_calls / self.attempts if self.attempts else float("nan")

    @property
    def solver_calls_per_sample(self) -> float:
        return self.solver_calls / self.accepted if self.accepted else float("nan")

    @property
    def optimality_rejection_rate(self) -> float:
        """Share of solved candidates discarded for failing local optimality."""
        solved = self.attempts - self.solver_failures - self.infeasible_rejections
        return self.optimality_rejections / solved if solved > 0 else 0.0

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_s / self.accepted if self.accepted else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state-bound",
        type=float,
        default=4.0,
        help="Symmetric sampling bound for both initial-state components.",
    )
    parser.add_argument(
        "--projection-weight",
        type=float,
        default=100.0,
        help="State-distance weight Qx in the nonlinear feasible-set projection.",
    )
    parser.add_argument(
        "--slack-penalty",
        choices=("quadratic", "l1"),
        default="quadratic",
        help="Penalty used by the slacked extreme-state sampler.",
    )
    parser.add_argument(
        "--close-to-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive the slacked sampler toward the bounds of the feasible set.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4,
        help="Closed-loop MPC steps simulated from every bilevel seed.",
    )
    parser.add_argument(
        "--rollout-seeds",
        type=int,
        default=0,
        help=(
            "Number of bilevel seeds for the rollout method. "
            "0 selects the smallest count that can supply --num-samples."
        ),
    )
    parser.add_argument(
        "--rollout-dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop a rollout when it revisits a state already collected, so the "
            "origin is not stored once per seed."
        ),
    )
    parser.add_argument(
        "--rollout-tol",
        type=float,
        default=ROLLOUT_ORIGIN_TOL,
        help="Stop a closed-loop rollout once the state is this close to the origin.",
    )
    parser.add_argument(
        "--linearization-points",
        type=int,
        default=4,
        help=(
            "Reference points whose linearized feasible set is compared with "
            "the true one. 0 disables that figure."
        ),
    )
    parser.add_argument(
        "--mpc-restarts",
        type=int,
        default=MPC_RESTARTS,
        help=(
            "Randomized MPC initial guesses tried after the rank-safe one "
            "before a state is reported infeasible."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Independent reruns (different seeds) used to report timing and "
            "acceptance rate as mean +/- std. The first run (--seed) also "
            "produces every figure and the feasibility/SOSC check, which are "
            "not repeated. 1 disables repetition (std is then undefined)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep solver/sampler progress output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "discontinuous_control_projection_sampling",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.horizon < 2:
        parser.error("--horizon must be at least 2")
    if not 0.0 < args.state_bound <= 4.0:
        parser.error("--state-bound must lie in (0, 4]")
    if args.projection_weight < 0.0:
        parser.error("--projection-weight must be non-negative")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.rollout_seeds < 0:
        parser.error("--rollout-seeds must be non-negative")
    if args.mpc_restarts < 0:
        parser.error("--mpc-restarts must be non-negative")
    if args.linearization_points < 0:
        parser.error("--linearization-points must be non-negative")
    if args.rollout_tol < 0.0:
        parser.error("--rollout-tol must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def feasible_interval(
    abscissa: np.ndarray, horizon: int, control_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(t_min, t_max)`` of the feasible set at each abscissa ``s``.

    See :func:`feasible_set_profile` for the derivation. Entries where the set
    is empty come back as ``(+inf, -inf)``.
    """
    N = int(horizon)
    U = float(control_bound)
    if N < 1 or U <= 0.0:
        raise ValueError("horizon must be positive and control_bound must be positive.")

    abscissa = np.asarray(abscissa, dtype=float)
    lower = np.full(abscissa.shape, np.inf)
    upper = np.full(abscissa.shape, -np.inf)
    tol = 1e-9

    for p in range(N + 1):
        for q in range(N + 1 - p):
            for m in range(N + 1 - p - q):
                n = N - p - q - m
                base_s = (p - q) * U
                base_t = (p - q) * U**3
                spread = m - n
                if spread == 0:
                    # The +/-a components cancel, so this family only reaches
                    # the single abscissa fixed by the saturated components.
                    admissible = np.abs(abscissa - base_s) <= tol
                    values = np.full(abscissa.shape, float(base_t))
                else:
                    a = (abscissa - base_s) / spread
                    admissible = (a >= -tol) & (a <= U + tol)
                    values = base_t + spread * np.clip(a, 0.0, U) ** 3
                lower = np.where(admissible, np.minimum(lower, values), lower)
                upper = np.where(admissible, np.maximum(upper, values), upper)

    return lower, upper


def feasible_set_profile(
    horizon: int, control_bound: float, num_points: int = FEASIBLE_SET_RESOLUTION
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(s, t_min(s), t_max(s))``: the exact MPC feasible set.

    With ``x_{k+1} = x_k + [u_k, u_k^3]`` and the terminal constraint
    ``x_N = 0``, a state is feasible exactly when
    ``-x_0 = (sum_k u_k, sum_k u_k^3)`` for some ``|u_k| <= U``. The set is
    symmetric under ``u -> -u``, so it is also the set of admissible
    ``(sum u_k, sum u_k^3)`` pairs itself.

    For a fixed ``s = sum_k u_k`` the admissible inputs form a convex set and
    ``sum_k u_k^3`` is continuous, so the reachable ``t`` values fill an
    interval. Its endpoints solve ``max/min sum u_k^3`` subject to
    ``sum u_k = s`` and ``|u_k| <= U``; the KKT conditions of that problem give
    ``3 u_k^2 = lambda`` for every interior component, so each ``u_k`` equals
    ``+/-U`` or ``+/-a`` for a single common ``a`` in ``[0, U]``. Enumerating the
    multiplicities (``p`` at ``+U``, ``q`` at ``-U``, ``m`` at ``+a``, ``n`` at
    ``-a``) therefore contains every optimizer, and each enumerated point is
    itself attainable -- so the extrema are exact rather than a local guess.
    """
    s_grid = np.linspace(-int(horizon) * float(control_bound),
                         int(horizon) * float(control_bound), num_points)
    lower, upper = feasible_interval(s_grid, horizon, control_bound)

    reachable = np.isfinite(lower) & np.isfinite(upper)
    return s_grid[reachable], lower[reachable], upper[reachable]


def zonotope_vertices(center: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """Return the vertices of a 2-D zonotope, ordered around its boundary.

    Every generator is flipped into the upper half plane and sorted by angle;
    walking the sorted generators forwards from the bottom vertex traces the
    upper chain and walking them backwards traces the lower one.
    """
    generators = np.array(generators, dtype=float)
    if generators.ndim != 2 or generators.shape[0] != 2:
        raise ValueError("generators must have shape (2, k).")
    flip = (generators[1] < 0.0) | ((generators[1] == 0.0) & (generators[0] < 0.0))
    generators[:, flip] *= -1.0
    order = np.argsort(np.arctan2(generators[1], generators[0]), kind="stable")
    generators = generators[:, order]

    point = np.asarray(center, dtype=float).reshape(2) - generators.sum(axis=1)
    vertices = [point]
    for generator in generators.T:
        vertices.append(vertices[-1] + 2.0 * generator)
    for generator in generators.T:
        vertices.append(vertices[-1] - 2.0 * generator)
    return np.array(vertices[:-1])


def linearized_feasible_set(model: DiscontinuousControl, solution: dict) -> np.ndarray:
    """Return the vertices of ``X_lin`` around an MPC solution.

    The model condenses its local problem to ``M du = -dx0`` with
    ``|u_nominal + du| <= U`` (see ``_linearized_opc_at_solution``), so

        X_lin = {x0_nominal - M (u - u_nominal) : |u| <= U},

    a zonotope centred at ``x0_nominal + M u_nominal`` whose generators are the
    columns of ``M`` scaled by the input bound. No state bound enters here,
    exactly as in the sampler's linearized problem.
    """
    local = model._linearized_opc_at_solution(solution["w"])
    M = np.asarray(local["M"], dtype=float)
    center = np.asarray(local["x0_nominal"], dtype=float) + M @ np.asarray(
        local["U_nominal"], dtype=float
    )
    return zonotope_vertices(center, model.control_bound * M)


def polygon_vertical_slice(vertices: np.ndarray, abscissa: float):
    """Return the ``[y_min, y_max]`` cut of a convex polygon at ``x = abscissa``."""
    ys: List[float] = []
    count = vertices.shape[0]
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        if x1 == x2:
            if x1 == abscissa:
                ys.extend((y1, y2))
            continue
        if min(x1, x2) - 1e-12 <= abscissa <= max(x1, x2) + 1e-12:
            ys.append(y1 + (y2 - y1) * (abscissa - x1) / (x2 - x1))
    if not ys:
        return None
    return min(ys), max(ys)


def set_overlap(
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    vertices: np.ndarray,
    model_bounds: np.ndarray,
    num_points: int = 2001,
) -> dict:
    """Compare ``X`` and ``X_lin`` by area, both clipped to the state box.

    Both sets are cut into vertical slices, which is exact here: ``X`` is an
    interval in every slice by construction and ``X_lin`` is convex.
    """
    s_values, lower, upper = profile
    grid = np.linspace(-model_bounds[0], model_bounds[0], num_points)
    true_low = np.clip(np.interp(grid, s_values, lower), -model_bounds[1], model_bounds[1])
    true_high = np.clip(np.interp(grid, s_values, upper), -model_bounds[1], model_bounds[1])
    outside = np.abs(grid) > s_values[-1]
    true_low[outside] = 0.0
    true_high[outside] = 0.0

    lin_low = np.zeros(num_points)
    lin_high = np.zeros(num_points)
    for index, abscissa in enumerate(grid):
        cut = polygon_vertical_slice(vertices, float(abscissa))
        if cut is None:
            continue
        lin_low[index] = np.clip(cut[0], -model_bounds[1], model_bounds[1])
        lin_high[index] = np.clip(cut[1], -model_bounds[1], model_bounds[1])

    true_length = np.maximum(true_high - true_low, 0.0)
    lin_length = np.maximum(lin_high - lin_low, 0.0)
    shared = np.maximum(
        np.minimum(true_high, lin_high) - np.maximum(true_low, lin_low), 0.0
    )
    area_true = float(np.trapezoid(true_length, grid))
    area_lin = float(np.trapezoid(lin_length, grid))
    area_shared = float(np.trapezoid(shared, grid))
    return {
        "area_true": area_true,
        "area_linearized": area_lin,
        "area_shared": area_shared,
        # Share of X_lin that is genuinely feasible: the rest needs projection.
        "valid_fraction": area_shared / area_lin if area_lin > 0.0 else float("nan"),
        # Share of X the linearization can reach at all from this point.
        "covered_fraction": area_shared / area_true if area_true > 0.0 else float("nan"),
    }


def select_reference_points(results: List[SamplingResult], count: int) -> np.ndarray:
    """Pick well-separated accepted samples to linearize around.

    A greedy farthest-point walk started from the most central sample mixes
    interior and boundary states, where the linearization behaves differently.
    """
    pool = [result.samples for result in results if result.samples.size]
    if not pool or count <= 0:
        return np.empty((0, 2), dtype=float)
    points = np.unique(np.vstack(pool), axis=0)
    chosen = [int(np.argmin(np.linalg.norm(points, axis=1)))]
    while len(chosen) < min(count, points.shape[0]):
        distances = np.min(
            np.linalg.norm(points[:, None, :] - points[chosen][None, :, :], axis=2),
            axis=1,
        )
        chosen.append(int(np.argmax(distances)))
    return points[chosen]


def draw_feasible_set(
    ax,
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_bounds: np.ndarray,
    label: bool = False,
) -> None:
    """Shade the feasible set, clipped to the state box of the MPC problem."""
    s_values, lower, upper = profile
    inside = np.abs(s_values) <= model_bounds[0] + 1e-12
    s_values = s_values[inside]
    lower = np.clip(lower[inside], -model_bounds[1], model_bounds[1])
    upper = np.clip(upper[inside], -model_bounds[1], model_bounds[1])
    nonempty = upper > lower

    ax.fill_between(
        s_values,
        lower,
        upper,
        where=nonempty,
        color="0.88",
        zorder=0,
        label="MPC feasible set" if label else None,
    )
    for boundary in (lower, upper):
        masked = np.where(nonempty, boundary, np.nan)
        ax.plot(s_values, masked, color="0.45", linewidth=1.0, zorder=0)


def draw_sampling_box(
    ax, bounds: np.ndarray, model_bounds: np.ndarray, label: bool = False
) -> None:
    """Outline the sampling box when it is tighter than the MPC state box."""
    if np.all(bounds >= model_bounds - 1e-12):
        return
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (-bounds[0], -bounds[1]),
            2.0 * bounds[0],
            2.0 * bounds[1],
            fill=False,
            linestyle="--",
            linewidth=1.1,
            color="0.25",
            zorder=1,
            label="sampling box" if label else None,
        )
    )


def save_feasible_set(
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_bounds: np.ndarray,
    output_dir: Path,
) -> None:
    """Store the analytic feasible-set boundary, clipped to the state box."""
    s_values, lower, upper = profile
    inside = np.abs(s_values) <= model_bounds[0] + 1e-12
    np.savetxt(
        output_dir / "feasible_set_boundary.csv",
        np.column_stack(
            (
                s_values[inside],
                np.clip(lower[inside], -model_bounds[1], model_bounds[1]),
                np.clip(upper[inside], -model_bounds[1], model_bounds[1]),
            )
        ),
        delimiter=",",
        header="x0_1,x0_2_min,x0_2_max",
        comments="",
    )


def inside_feasible_set(
    model: DiscontinuousControl,
    samples: np.ndarray,
    model_bounds: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """Exact membership test against the analytic feasible set.

    The interval is evaluated at each sample's own abscissa rather than
    interpolated from the plotting grid: samplers that aim at the boundary put
    their samples exactly on the curve, where a chord between grid points would
    cut the corner and reject feasible states.
    """
    samples = normalize_samples(samples)
    if samples.size == 0:
        return np.zeros(0, dtype=bool)
    within_box = np.all(np.abs(samples) <= model_bounds + tol, axis=1)
    lower_at, upper_at = feasible_interval(
        samples[:, 0], model.N, model.control_bound
    )
    within_interval = (samples[:, 1] >= lower_at - tol) & (samples[:, 1] <= upper_at + tol)
    return within_box & within_interval


def normalize_samples(samples: object) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected samples with shape (n, 2), got {values.shape}.")
    return values


def quiet(verbose: bool):
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


def mpc_warm_starts(
    model: DiscontinuousControl, x0: np.ndarray, restarts: int, rng: np.random.Generator
):
    """Yield MPC initial guesses that keep the constraint Jacobian full rank.

    The model's stored guess sets every input to zero, which is exactly the
    degenerate point of this dynamics: ``d(sum u_k^3)/du_k = 3 u_k^2`` vanishes
    at ``u = 0``, so the second terminal-constraint row loses its dependence on
    the free variables (``x_0`` is already pinned by ``x_0 = p``). LICQ then
    fails at the initial guess and IPOPT reports
    ``Infeasible_Problem_Detected`` for perfectly feasible states. Every guess
    below has nonzero inputs.
    """
    N = model.N
    # A zero-sum perturbation keeps sum(u) at the value the linear part of the
    # terminal constraint needs while moving every component away from zero.
    level = -float(x0[0]) / N
    alternating = 0.5 * model.control_bound * (-1.0) ** np.arange(N)
    yield np.r_[x0, np.clip(level + alternating, -model.control_bound, model.control_bound)]
    for _ in range(max(restarts, 0)):
        yield np.r_[
            x0,
            rng.uniform(-0.75 * model.control_bound, 0.75 * model.control_bound, N),
        ]


class SolverCalls:
    """Counts NLP solves.

    An attempt is one try at producing a feasible and optimal sample; it is not
    the same as one solver call. A retried MPC solve costs several calls, and a
    projected candidate costs its generator solve plus the projection solve, so
    the two quantities are tracked separately.
    """

    def __init__(self) -> None:
        self.count = 0

    def add(self, calls: int = 1) -> None:
        self.count += calls


def solve_mpc(
    model: DiscontinuousControl,
    x0: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    restarts: Optional[int] = None,
    tol: float = 1e-6,
    calls: Optional[SolverCalls] = None,
) -> Optional[dict]:
    """Solve the original MPC at ``x0``, retrying from fresh guesses.

    Returns the primal solution, the equality multipliers, the bound
    multipliers and the number of extra restarts needed, or ``None`` when no
    guess produced a solution that satisfies the constraints.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape != (2,) or not np.all(np.isfinite(x0)):
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    restarts = MPC_RESTARTS if restarts is None else restarts

    for attempt, guess in enumerate(mpc_warm_starts(model, x0, restarts, rng)):
        if calls is not None:
            calls.add()
        try:
            solution = model.solver(
                x0=guess,
                lbx=model.lbw,
                ubx=model.ubw,
                lbg=model.lbg,
                ubg=model.ubg,
                p=x0,
            )
        except Exception:
            continue
        if not model.solver.stats().get("success", False):
            continue

        w = solution["x"].full().reshape(-1)
        solved_x0, controls = model.extract_traj(solution["x"])
        solved_x0 = solved_x0.full().reshape(2)
        controls = controls.full().reshape(model.N)
        terminal = solved_x0 + np.array([np.sum(controls), np.sum(controls**3)])
        if (
            np.linalg.norm(solved_x0 - x0, ord=np.inf) > tol
            or np.linalg.norm(terminal, ord=np.inf) > tol
            or np.any(np.abs(controls) > model.control_bound + tol)
        ):
            continue

        return {
            "w": w,
            "x0": solved_x0,
            "controls": controls,
            "lam_h": solution["lam_g"].full().reshape(-1),
            "lam_x": solution["lam_x"].full().reshape(-1),
            "restarts": attempt,
        }
    return None


def solve_candidate(
    model: DiscontinuousControl,
    candidate: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    calls: Optional[SolverCalls] = None,
) -> Tuple[bool, float]:
    """Return MPC feasibility of ``candidate`` and its optimal first input."""
    solution = solve_mpc(model, candidate, rng=rng, calls=calls)
    if solution is None:
        return False, float("nan")
    return True, float(solution["controls"][0])


def check_sosc(
    model: DiscontinuousControl, solution: dict, tol: float = 1e-6
) -> dict:
    """Check the second-order sufficient conditions at an MPC solution.

    ``Z`` spans the null space of the equality Jacobian stacked with the
    Jacobians of the active bound constraints, and SOSC holds when the reduced
    Hessian ``Z' * H * Z`` is positive definite. The bound constraints are
    linear, so their multipliers do not enter ``H`` and only the equality
    multipliers are needed.

    An active constraint whose multiplier vanishes makes ``Z`` a strict subset
    of the critical cone, which turns the test conservative; those cases are
    counted in ``degenerate`` so they stay visible.
    """
    w = solution["w"]
    residual = model.g(w).full().reshape(-1)
    active = residual >= -tol

    jac_h = model.jac_h(solution["x0"], w).full()
    jac_g = model.jac_g(w).full()
    active_jacobian = np.vstack([jac_h, jac_g[active, :]]) if np.any(active) else jac_h

    # Each bound row of g touches exactly one variable, so its multiplier is
    # the corresponding IPOPT bound multiplier, signed by the row's direction.
    lam_x = solution["lam_x"]
    degenerate = 0
    for row in np.flatnonzero(active):
        column = int(np.argmax(np.abs(jac_g[row, :])))
        if abs(lam_x[column]) <= tol:
            degenerate += 1

    null_basis = null_space(active_jacobian)
    if null_basis.size == 0:
        # No feasible direction left: the point is a locally unique solution.
        return {
            "holds": True,
            "min_eigenvalue": float("inf"),
            "reduced_dimension": 0,
            "degenerate_active": degenerate,
        }

    lam_g = np.zeros(model.g(w).numel())
    hessian = model.hessian_L(w, solution["lam_h"], lam_g).full()
    reduced = null_basis.T @ hessian @ null_basis
    reduced = 0.5 * (reduced + reduced.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(reduced)))
    return {
        "holds": bool(min_eigenvalue > tol),
        "min_eigenvalue": min_eigenvalue,
        "reduced_dimension": int(null_basis.shape[1]),
        "degenerate_active": degenerate,
    }


def mpc_first_control(
    model: DiscontinuousControl,
    state: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    calls: Optional[SolverCalls] = None,
) -> Optional[float]:
    """Return the optimal first input at ``state``, or None if the MPC fails.

    ``DiscontinuousControl.solve_MPC`` returns ``sol['x'][0]``, which is the
    first *state* component of the decision vector ``[x0, u_0, ..., u_{N-1}]``,
    so the control is extracted through :func:`solve_mpc` instead.
    """
    solution = solve_mpc(model, state, rng=rng, calls=calls)
    return None if solution is None else float(solution["controls"][0])


def project_candidates(
    model: DiscontinuousControl,
    candidates: np.ndarray,
    projection_weight: float,
    num_samples: int,
    calls: Optional[SolverCalls] = None,
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """Project candidates onto the nonlinear MPC feasible set.

    The origin is the projection warm start: it is feasible for every horizon
    because the zero input sequence already satisfies the terminal constraint.
    It is solved once here and handed to every projection, because
    ``project_onto_feasible_set`` otherwise re-solves that same MPC on every
    call. Each candidate then costs exactly one projection solve.
    """
    projected: List[np.ndarray] = []
    shifts: List[float] = []
    moved = 0
    failures = 0

    reference_solution = model.solver(
        x0=model.w0,
        lbx=model.lbw,
        ubx=model.ubw,
        lbg=model.lbg,
        ubg=model.ubg,
        p=np.zeros(2),
    )
    if calls is not None:
        calls.add()

    for candidate in candidates:
        if len(projected) >= num_samples:
            break
        if calls is not None:
            calls.add()
        try:
            point = model.project_onto_feasible_set(
                candidate,
                Qx=projection_weight,
                reference_solution=reference_solution,
            )
        except RuntimeError:
            failures += 1
            continue
        shift = float(np.linalg.norm(point - candidate))
        shifts.append(shift)
        if shift > PROJECTION_MOVE_TOL:
            moved += 1
        projected.append(point)

    return (
        normalize_samples(projected),
        moved,
        failures,
        np.asarray(shifts, dtype=float),
    )


def collect_uniform(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    model_bounds: np.ndarray,
) -> SamplingResult:
    rng = np.random.default_rng(seed)
    solver_rng = np.random.default_rng(seed + 1)
    samples: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    accepted_mask: List[bool] = []
    rejected: List[np.ndarray] = []
    calls = SolverCalls()

    start = perf_counter()
    for _ in range(max_attempts):
        candidate = rng.uniform(-bounds, bounds)
        candidates.append(candidate)
        feasible, _ = solve_candidate(model, candidate, rng=solver_rng, calls=calls)
        accepted_mask.append(feasible)
        if feasible:
            samples.append(candidate)
        else:
            rejected.append(candidate)
        if len(samples) >= num_samples:
            break
    elapsed = perf_counter() - start

    # The analytic set separates a genuinely infeasible draw from a state the
    # solver merely failed on, which a pass/fail MPC check cannot distinguish.
    rejected_array = normalize_samples(rejected)
    if rejected_array.size:
        outside = ~inside_feasible_set(model, rejected_array, model_bounds)
        infeasible = int(np.sum(outside))
    else:
        infeasible = 0

    return SamplingResult(
        method="uniform",
        samples=normalize_samples(samples),
        elapsed_s=elapsed,
        generation_s=elapsed,
        attempts=len(candidates),
        candidates=normalize_samples(candidates),
        candidate_feasible=np.asarray(accepted_mask, dtype=bool),
        solver_calls=calls.count,
        infeasible_rejections=infeasible,
        solver_failures=int(rejected_array.shape[0]) - infeasible,
    )


def collect_bilevel_projection(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    projection_weight: float,
    verbose: bool,
) -> SamplingResult:
    # The bilevel residual-weight sampler draws from NumPy's legacy global RNG.
    np.random.seed(seed)
    start = perf_counter()
    try:
        with quiet(verbose):
            candidates = model.collect_feasible_samples_with_bilevel(
                x_max=bounds,
                num_samples=num_samples,
                max_attempts=max_attempts,
                plot_results=False,
            )
    except Exception as exc:
        raise RuntimeError(
            "The bilevel sampler failed. It needs the CasADi ccopt plugin."
        ) from exc
    generation_s = perf_counter() - start
    candidates = normalize_samples(candidates)
    stats = model.last_feasible_sampling_stats
    attempts = int(stats["attempts"])
    # The sampler solves, then screens each solution with a reduced-Hessian
    # (SOSC) test, so its two loss channels are reported separately.
    solved = int(stats["feasible_candidates"])
    optimal = int(stats["feasible_and_optimal_candidates"])
    # The sampler calls its ccopt solver exactly once per attempt.
    calls = SolverCalls()
    calls.add(attempts)

    start = perf_counter()
    with quiet(verbose):
        samples, moved, failures, shifts = project_candidates(
            model, candidates, projection_weight, num_samples, calls=calls
        )
    projection_s = perf_counter() - start

    return SamplingResult(
        method="bilevel_projection",
        solver_calls=calls.count,
        solver_failures=attempts - solved,
        optimality_rejections=solved - optimal,
        samples=samples,
        elapsed_s=generation_s + projection_s,
        generation_s=generation_s,
        post_processing_s=projection_s,
        attempts=attempts,
        candidates=candidates,
        projection_attempts=int(candidates.shape[0]),
        projection_moved=moved,
        projection_failures=failures,
        projection_shifts=shifts,
    )


def collect_slacked_projection(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    projection_weight: float,
    slack_penalty: str,
    close_to_bounds: bool,
    verbose: bool,
) -> SamplingResult:
    # The spiked residual weights also come from the legacy global RNG, so the
    # same seed makes this sampler reproducible next to the bilevel one.
    np.random.seed(seed)
    start = perf_counter()
    with quiet(verbose):
        candidates = model.collect_feasible_samples_with_slacked(
            x_max=bounds,
            close_to_bounds=close_to_bounds,
            num_samples=num_samples,
            max_attempts=max_attempts,
            plot_results=False,
            slack_penalty=slack_penalty,
        )
    generation_s = perf_counter() - start
    candidates = normalize_samples(candidates)
    stats = model.last_feasible_sampling_stats
    attempts = int(stats["attempts"])
    # The slacked sampler has no optimality screen of its own: every solved
    # extreme-state problem is kept, so local optimality is only measured
    # afterwards by the SOSC check in verify_results.
    solved = int(stats["successful_extreme_solves"])
    # Each attempt is one seek_x0 call, which solves its NLP exactly once.
    calls = SolverCalls()
    calls.add(attempts)

    start = perf_counter()
    with quiet(verbose):
        samples, moved, failures, shifts = project_candidates(
            model, candidates, projection_weight, num_samples, calls=calls
        )
    projection_s = perf_counter() - start

    return SamplingResult(
        method="slacked_projection",
        solver_calls=calls.count,
        solver_failures=attempts - solved,
        samples=samples,
        elapsed_s=generation_s + projection_s,
        generation_s=generation_s,
        post_processing_s=projection_s,
        attempts=attempts,
        candidates=candidates,
        projection_attempts=int(candidates.shape[0]),
        projection_moved=moved,
        projection_failures=failures,
        projection_shifts=shifts,
    )


def collect_bilevel_rollout(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    model_bounds: np.ndarray,
    seed: int,
    rollout_steps: int,
    rollout_seeds: int,
    verbose: bool,
    rollout_tol: float = ROLLOUT_ORIGIN_TOL,
    dedupe: bool = True,
) -> SamplingResult:
    """Propagate bilevel seeds with closed-loop MPC and keep every visited state.

    Each rollout contributes its seed plus the states reached under the MPC
    feedback law, so a boundary seed also populates the interior of the
    feasible set. A rollout stops early when

    * the successor leaves the state box. The MPC constrains only ``x_0``, not
      its predicted trajectory, so the closed loop is not box invariant: with
      ``|u| <= U`` the successor of a boundary state can reach ``|x_2| = U^3``.
      Such a state is not an admissible MPC initial state, so it is dropped
      rather than stored;
    * the state has reached the origin, where every further step would only
      duplicate it;
    * the state repeats one already collected. The dynamics are deterministic,
      so the rest of that trajectory would repeat too. Every rollout ends at
      the origin, so without this check the origin would be stored once per
      seed;
    * the MPC solve fails.

    Because a rollout can be shorter than ``rollout_steps + 1``, seeds are
    drawn in batches until ``num_samples`` states exist. Passing an explicit
    ``rollout_seeds`` disables that top-up and uses exactly that many seeds.
    """
    states_per_rollout = rollout_steps + 1
    explicit_seeds = rollout_seeds > 0

    np.random.seed(seed)
    solver_rng = np.random.default_rng(seed + 1)
    samples: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []
    seed_states: List[np.ndarray] = []
    generation_s = 0.0
    rollout_s = 0.0
    attempts = 0
    solver_failures = 0
    optimality_rejections = 0
    simulated_steps = 0
    rollout_failures = 0
    left_box = 0
    converged = 0
    duplicates = 0
    batches = 0
    calls = SolverCalls()
    batch_yield: Optional[float] = None
    seen: set = set()

    def register(state: np.ndarray) -> bool:
        """Record a state, reporting whether it is new."""
        if not dedupe:
            return True
        key = tuple(np.round(state, 9))
        if key in seen:
            return False
        seen.add(key)
        return True
    remaining_attempts = max_attempts

    while len(samples) < num_samples and remaining_attempts > 0 and batches < ROLLOUT_MAX_BATCHES:
        missing = num_samples - len(samples)
        if explicit_seeds:
            requested = rollout_seeds
        elif batch_yield is not None:
            # Size the top-up from the marginal yield of the previous batch,
            # not from the cumulative average: as states accumulate, later
            # rollouts hit an already-collected state sooner, so the average
            # over-estimates what the next batch will return and every top-up
            # falls short again.
            requested = int(np.ceil(ROLLOUT_SEED_MARGIN * missing / batch_yield))
        else:
            requested = int(np.ceil(missing / states_per_rollout))
        requested = max(requested, 1)
        batch_start_samples = len(samples)
        batch_start_seeds = len(seed_states)

        start = perf_counter()
        try:
            with quiet(verbose):
                seeds = model.collect_feasible_samples_with_bilevel(
                    x_max=bounds,
                    num_samples=requested,
                    max_attempts=remaining_attempts,
                    plot_results=False,
                )
        except Exception as exc:
            raise RuntimeError(
                "The bilevel sampler failed. It needs the CasADi ccopt plugin."
            ) from exc
        generation_s += perf_counter() - start
        batches += 1

        seeds = normalize_samples(seeds)
        stats = model.last_feasible_sampling_stats
        batch_attempts = int(stats["attempts"])
        attempts += batch_attempts
        remaining_attempts -= batch_attempts
        # One ccopt solve per seed attempt.
        calls.add(batch_attempts)
        solved = int(stats["feasible_candidates"])
        solver_failures += batch_attempts - solved
        optimality_rejections += solved - int(stats["feasible_and_optimal_candidates"])
        if seeds.shape[0] == 0:
            break
        seed_states.extend(seeds)

        start = perf_counter()
        for seed_state in seeds:
            if len(samples) >= num_samples:
                break
            # States are stored as they are simulated, so no closed-loop step is
            # paid for and then thrown away by the sample-count truncation.
            if not register(seed_state):
                duplicates += 1
                continue
            trajectory = [seed_state]
            samples.append(seed_state)
            state = seed_state
            for _ in range(rollout_steps):
                if len(samples) >= num_samples:
                    break
                if np.max(np.abs(state)) <= rollout_tol:
                    converged += 1
                    break
                control = mpc_first_control(
                    model, state, rng=solver_rng, calls=calls
                )
                simulated_steps += 1
                if control is None:
                    rollout_failures += 1
                    break
                state = np.asarray(
                    model.step(state, [control]).full(), dtype=float
                ).reshape(2)
                if np.any(np.abs(state) > model_bounds + 1e-6):
                    left_box += 1
                    break
                if not register(state):
                    duplicates += 1
                    break
                trajectory.append(state)
                samples.append(state)
            trajectories.append(np.asarray(trajectory, dtype=float))
        rollout_s += perf_counter() - start

        new_seeds = len(seed_states) - batch_start_seeds
        if new_seeds > 0:
            batch_yield = max(
                (len(samples) - batch_start_samples) / new_seeds, 0.1
            )

        if explicit_seeds:
            break

    return SamplingResult(
        method="bilevel_rollout",
        samples=normalize_samples(samples),
        solver_calls=calls.count,
        elapsed_s=generation_s + rollout_s,
        generation_s=generation_s,
        post_processing_s=rollout_s,
        # Bilevel seed attempts plus every simulated closed-loop step.
        attempts=attempts + simulated_steps,
        candidates=normalize_samples(seed_states),
        # Seeds lost to the ccopt solver or to a failed closed-loop solve.
        solver_failures=solver_failures + rollout_failures,
        optimality_rejections=optimality_rejections,
        rollout_seeds=len(seed_states),
        rollout_steps=rollout_steps,
        rollout_failures=rollout_failures,
        rollout_left_box=left_box,
        rollout_converged=converged,
        rollout_duplicates=duplicates,
        rollout_batches=batches,
        trajectories=trajectories,
    )


def verify_results(
    model: DiscontinuousControl,
    results: List[SamplingResult],
    model_bounds: np.ndarray,
) -> None:
    """Recompute controls and check feasibility and local optimality.

    Runs after all timings, so neither the SOSC test nor the extra MPC solves
    bias the measured wall times. The same test is applied to every method,
    including the samplers that do not screen for optimality themselves.
    """
    rng = np.random.default_rng(0)
    for result in results:
        calls = SolverCalls()
        controls = np.full(result.accepted, np.nan)
        feasible_count = 0
        optimal_count = 0
        degenerate_count = 0
        restarts = 0
        for index, sample in enumerate(result.samples):
            solution = solve_mpc(model, sample, rng=rng, calls=calls)
            if solution is None:
                continue
            controls[index] = float(solution["controls"][0])
            feasible_count += 1
            restarts += int(solution["restarts"])
            sosc = check_sosc(model, solution)
            optimal_count += int(sosc["holds"])
            degenerate_count += int(sosc["degenerate_active"] > 0)
        result.controls = controls
        result.verified_feasible = feasible_count
        result.locally_optimal = optimal_count
        result.sosc_degenerate = degenerate_count
        result.mpc_restarts = restarts
        result.verification_solver_calls = calls.count
        result.analytic_feasible = int(
            np.sum(inside_feasible_set(model, result.samples, model_bounds))
        )
        if result.analytic_feasible > feasible_count:
            # An MPC solve can still fail on a point that is provably feasible,
            # so report the gap instead of silently trusting the solver.
            print(
                f"note: {LABELS[result.method]} has "
                f"{result.analytic_feasible - feasible_count} sample(s) inside the "
                "feasible set that the MPC solver did not confirm."
            )


def statistics_row(result: SamplingResult, requested: int) -> dict:
    samples = result.samples
    radii = np.linalg.norm(samples, axis=1) if samples.size else np.empty(0)
    shifts = result.projection_shifts
    return {
        "method": result.method,
        "label": LABELS[result.method],
        "requested": requested,
        "accepted": result.accepted,
        "verified_feasible": result.verified_feasible,
        "analytic_feasible": result.analytic_feasible,
        "locally_optimal": result.locally_optimal,
        "sosc_degenerate": result.sosc_degenerate,
        "attempts": result.attempts,
        "solver_calls": result.solver_calls,
        "solver_calls_per_attempt": result.solver_calls_per_attempt,
        "solver_calls_per_accepted_sample": result.solver_calls_per_sample,
        "verification_solver_calls": result.verification_solver_calls,
        "rejected": result.rejected,
        "rejection_rate": result.rejection_rate,
        "infeasible_rejections": result.infeasible_rejections,
        "solver_failures": result.solver_failures,
        "optimality_rejections": result.optimality_rejections,
        "optimality_rejection_rate": result.optimality_rejection_rate,
        "verification_mpc_restarts": result.mpc_restarts,
        "computation_time_s": result.elapsed_s,
        "generation_time_s": result.generation_s,
        "post_processing_time_s": result.post_processing_s,
        "seconds_per_accepted_sample": result.seconds_per_sample,
        "projection_attempts": result.projection_attempts,
        "projection_moved": result.projection_moved,
        "projection_failures": result.projection_failures,
        "mean_projection_shift": float(np.mean(shifts)) if shifts.size else float("nan"),
        "max_projection_shift": float(np.max(shifts)) if shifts.size else float("nan"),
        "rollout_seeds": result.rollout_seeds,
        "rollout_steps": result.rollout_steps,
        "rollout_failures": result.rollout_failures,
        "rollout_left_box": result.rollout_left_box,
        "rollout_converged": result.rollout_converged,
        "rollout_duplicates": result.rollout_duplicates,
        "rollout_batches": result.rollout_batches,
        "unique_samples": int(np.unique(samples, axis=0).shape[0]) if samples.size else 0,
        "mean_state_radius": float(np.mean(radii)) if radii.size else float("nan"),
        "max_state_radius": float(np.max(radii)) if radii.size else float("nan"),
    }


def save_results(
    results: List[SamplingResult],
    rows: List[dict],
    args: argparse.Namespace,
) -> None:
    output_dir = args.output_dir
    for result in results:
        np.savetxt(
            output_dir / f"samples_{result.method}.csv",
            np.column_stack((result.samples, result.controls)),
            delimiter=",",
            header="x0_1,x0_2,u0_opt",
            comments="",
        )
        if result.candidates is not None and result.candidates.size:
            np.savetxt(
                output_dir / f"candidates_{result.method}.csv",
                result.candidates,
                delimiter=",",
                header="x0_1,x0_2",
                comments="",
            )

    rollout = next(
        (result for result in results if result.method == "bilevel_rollout"), None
    )
    if rollout is not None and rollout.trajectories:
        rows_out = [
            np.column_stack(
                (
                    np.full(traj.shape[0], index),
                    np.arange(traj.shape[0]),
                    traj,
                )
            )
            for index, traj in enumerate(rollout.trajectories)
        ]
        np.savetxt(
            output_dir / "rollout_trajectories.csv",
            np.vstack(rows_out),
            delimiter=",",
            header="rollout,step,x_1,x_2",
            comments="",
        )

    with (output_dir / "sampling_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "configuration": {
            "num_samples": args.num_samples,
            "max_attempts": args.max_attempts,
            "horizon": args.horizon,
            "seed": args.seed,
            "state_bound": args.state_bound,
            "projection_weight": args.projection_weight,
            "slack_penalty": args.slack_penalty,
            "close_to_bounds": args.close_to_bounds,
            "rollout_steps": args.rollout_steps,
            "rollout_seeds": args.rollout_seeds,
            "mpc_restarts": args.mpc_restarts,
            "linearization_points": args.linearization_points,
            "rollout_tol": args.rollout_tol,
            "rollout_dedupe": args.rollout_dedupe,
            "repeats": args.repeats,
        },
        "rejection_rate_definition": (
            "rejected / attempts, where rejected is split by cause: "
            "infeasible_rejections (the candidate lies outside the feasible "
            "set, decided with the analytic set rather than a solver verdict), "
            "solver_failures (a feasible candidate the solver could not "
            "solve), optimality_rejections (a solved candidate whose "
            "reduced-Hessian/SOSC screen failed, so it is not a strict local "
            "minimizer) and projection_failures. The bilevel sampler cannot "
            "produce an infeasible point by construction, so its only loss "
            "channels are solver failures and optimality rejections."
        ),
        "attempt_definition": (
            "An attempt is one try at producing a feasible and optimal sample: "
            "one uniform draw, one bilevel or slacked generator solve, or one "
            "closed-loop rollout step. solver_calls counts NLP solves instead, "
            "so the two differ whenever a try costs more than one solve: an MPC "
            "solve retried from a fresh initial guess (up to 1 + mpc_restarts "
            "calls) and a projected candidate, which pays its generator solve "
            "plus one projection solve. verification_solver_calls covers the "
            "post-hoc feasibility and SOSC pass and is excluded from the timed "
            "totals."
        ),
        "local_optimality_definition": (
            "locally_optimal counts accepted samples whose MPC solution "
            "satisfies the second-order sufficient conditions: the reduced "
            "Hessian Z' * hessian_L * Z is positive definite, with Z spanning "
            "the null space of the equality Jacobian stacked with the active "
            "bound Jacobians. sosc_degenerate counts samples with an active "
            "constraint carrying a zero multiplier, where that null space is a "
            "strict subset of the critical cone and the test is conservative."
        ),
        "optimality_notes": {
            "uniform": "Accepted after solving the original MPC with IPOPT.",
            "bilevel_projection": (
                "Bilevel KKT/reduced-Hessian screen, then nonlinear "
                "feasible-set projection."
            ),
            "slacked_projection": (
                "Randomly spiked slack weights push states to the bounds, then "
                "the nonlinear feasible-set projection repairs them."
            ),
            "bilevel_rollout": (
                "Bilevel seeds propagated with closed-loop MPC. The MPC bounds "
                "only x_0, not its predicted trajectory, so a successor can "
                "leave the state box and stop the rollout (rollout_left_box); "
                "such states are not admissible initial states and are dropped."
            ),
        },
        "statistics": rows,
    }
    with (output_dir / "sampling_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def plot_samples(
    results: List[SamplingResult],
    bounds: np.ndarray,
    model_bounds: np.ndarray,
    control_bound: float,
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
) -> None:
    if control_bound <= 0.0:
        raise ValueError("control_bound must be positive.")

    # A dedicated colorbar axis keeps the data axes equally sized; letting
    # fig.colorbar steal space from the last panel misaligns the shared axes.
    num_panels = len(results)
    fig = plt.figure(figsize=(4.0 * num_panels + 1.2, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(
        1,
        num_panels + 1,
        width_ratios=[1.0] * num_panels + [0.045],
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(num_panels)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])
    colorbar_axis = fig.add_subplot(grid[0, num_panels])

    # Use the physical input limits, rather than the extrema of one particular
    # sample batch, so a given color always denotes the same control value.
    normalization = matplotlib.colors.Normalize(vmin=-control_bound, vmax=control_bound)
    colormap = plt.get_cmap("coolwarm")
    for index, (ax, result) in enumerate(zip(axes, results)):
        draw_feasible_set(ax, profile, model_bounds, label=index == 0)
        draw_sampling_box(ax, bounds, model_bounds, label=index == 0)
        if result.candidate_feasible is not None and result.candidates is not None:
            # Rejected draws stay visible: they are what the acceptance rate of
            # a rejection sampler actually costs.
            discarded = result.candidates[~result.candidate_feasible]
            if discarded.size:
                ax.scatter(
                    discarded[:, 0],
                    discarded[:, 1],
                    s=22,
                    color="0.72",
                    alpha=0.9,
                    edgecolors="none",
                    zorder=2,
                    label="rejected (infeasible)",
                )
        if result.samples.size:
            ax.scatter(
                result.samples[:, 0],
                result.samples[:, 1],
                c=result.controls,
                cmap=colormap,
                norm=normalization,
                s=30,
                alpha=0.78,
                edgecolors="none",
                zorder=3,
            )
        ax.set_title(f"{LABELS[result.method]}\n$n={result.accepted}$")
        ax.set_xlim(-1.04 * bounds[0], 1.04 * bounds[0])
        ax.set_ylim(-1.04 * bounds[1], 1.04 * bounds[1])
        ax.set_xlabel(r"$x_{0,1}$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$x_{0,2}$")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=8)

    colorbar_mappable = matplotlib.cm.ScalarMappable(norm=normalization, cmap=colormap)
    colorbar_mappable.set_array([])
    fig.colorbar(
        colorbar_mappable,
        cax=colorbar_axis,
        label=r"first optimal control $u_0^\star$",
    )
    fig.savefig(output_dir / "sampled_initial_states.png", dpi=190)
    fig.savefig(output_dir / "sampled_initial_states.pdf", bbox_inches="tight")


def plot_post_processing(
    results: List[SamplingResult],
    bounds: np.ndarray,
    model_bounds: np.ndarray,
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
) -> None:
    """Show what projection and rollouts do to the generated candidates."""
    projection_results = [
        result for result in results if result.projection_attempts and result.candidates is not None
    ]
    rollout = next(
        (result for result in results if result.method == "bilevel_rollout"), None
    )
    panels = projection_results + ([rollout] if rollout is not None else [])
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.0), squeeze=False)
    axes = axes.ravel()
    for ax, result in zip(axes, panels):
        draw_feasible_set(ax, profile, model_bounds, label=True)
        draw_sampling_box(ax, bounds, model_bounds, label=True)
        if result.method == "bilevel_rollout":
            for trajectory in result.trajectories:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color="0.6",
                    linewidth=0.9,
                    zorder=1,
                )
            if result.candidates is not None and result.candidates.size:
                ax.scatter(
                    result.candidates[:, 0],
                    result.candidates[:, 1],
                    s=34,
                    color="tab:red",
                    label="bilevel seeds",
                    zorder=3,
                )
            if result.samples.size:
                ax.scatter(
                    result.samples[:, 0],
                    result.samples[:, 1],
                    s=16,
                    color="tab:orange",
                    alpha=0.75,
                    label="rollout states",
                    zorder=2,
                )
            ax.set_title(
                f"{LABELS[result.method]}\n"
                f"{result.rollout_seeds} seeds x {result.rollout_steps} closed-loop steps"
            )
        else:
            candidates = result.candidates
            paired = min(candidates.shape[0], result.samples.shape[0])
            for index in range(paired):
                ax.plot(
                    (candidates[index, 0], result.samples[index, 0]),
                    (candidates[index, 1], result.samples[index, 1]),
                    color="0.6",
                    linewidth=0.8,
                    zorder=1,
                )
            ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                s=22,
                color="0.35",
                alpha=0.7,
                label="raw candidates",
                zorder=2,
            )
            if result.samples.size:
                ax.scatter(
                    result.samples[:, 0],
                    result.samples[:, 1],
                    s=26,
                    color=COLORS[result.method],
                    alpha=0.85,
                    label="projected",
                    zorder=3,
                )
            moved_fraction = (
                result.projection_moved / result.projection_attempts
                if result.projection_attempts
                else float("nan")
            )
            ax.set_title(
                f"{LABELS[result.method]}\nprojection moved "
                f"{100.0 * moved_fraction:.0f}% of candidates"
            )
        ax.set_xlim(-1.04 * bounds[0], 1.04 * bounds[0])
        ax.set_ylim(-1.04 * bounds[1], 1.04 * bounds[1])
        ax.set_xlabel(r"$x_{0,1}$")
        ax.set_ylabel(r"$x_{0,2}$")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "post_processing_effect.png", dpi=190)
    fig.savefig(output_dir / "post_processing_effect.pdf", bbox_inches="tight")


def linearization_report(
    model: DiscontinuousControl,
    references: np.ndarray,
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_bounds: np.ndarray,
) -> List[dict]:
    """Build ``X_lin`` and its overlap with ``X`` at each reference point."""
    rng = np.random.default_rng(0)
    entries: List[dict] = []
    for reference in references:
        solution = solve_mpc(model, reference, rng=rng)
        if solution is None:
            continue
        vertices = linearized_feasible_set(model, solution)
        entry = {
            "reference": np.asarray(reference, dtype=float),
            "controls": solution["controls"],
            "vertices": vertices,
        }
        entry.update(set_overlap(profile, vertices, model_bounds))
        # The generators are the columns of M = [1...1; 3u_1^2...3u_N^2], so the
        # zonotope collapses onto a segment whenever the u_k^2 are all equal --
        # in particular near u = 0, the same degeneracy that makes the terminal
        # constraint Jacobian lose rank there.
        entry["degenerate"] = bool(
            entry["area_linearized"] < 1e-3 * max(entry["area_true"], 1e-12)
        )
        entries.append(entry)
    return entries


def plot_linearized_sets(
    entries: List[dict],
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_bounds: np.ndarray,
    output_dir: Path,
) -> None:
    """Compare the true feasible set with the linearized one, point by point."""
    if not entries:
        return

    columns = len(entries)
    fig, grid_axes = plt.subplots(
        2, columns, figsize=(4.6 * columns, 9.0), squeeze=False
    )
    for column, entry in enumerate(entries):
        for row in range(2):
            _draw_linearized_panel(
                grid_axes[row, column],
                entry,
                profile,
                model_bounds,
                zoom=row == 1,
            )

    fig.suptitle(
        "True feasible set versus the linearized set at individual reference "
        "points\n(top: full extent of $\\mathcal{X}_{\\mathrm{lin}}$, "
        "bottom: zoom on the state box, where the overlap is measured)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "linearized_feasible_set.png", dpi=190)
    fig.savefig(output_dir / "linearized_feasible_set.pdf", bbox_inches="tight")


def _draw_linearized_panel(
    ax,
    entry: dict,
    profile: Tuple[np.ndarray, np.ndarray, np.ndarray],
    model_bounds: np.ndarray,
    zoom: bool,
) -> None:
    """Draw one reference point, either at full extent or zoomed to the box."""
    vertices = entry["vertices"]
    closed = np.vstack([vertices, vertices[:1]])

    draw_feasible_set(ax, profile, model_bounds, label=True)
    # The linearized problem carries no state bound, so show the raw
    # zonotope as well as the part the MPC box would actually admit.
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="tab:purple",
        linewidth=1.6,
        linestyle="--",
        zorder=4,
        label=r"$\mathcal{X}_{\mathrm{lin}}$",
    )
    ax.fill(
        closed[:, 0],
        closed[:, 1],
        color="tab:purple",
        alpha=0.12,
        zorder=1,
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (-model_bounds[0], -model_bounds[1]),
            2.0 * model_bounds[0],
            2.0 * model_bounds[1],
            fill=False,
            linestyle=":",
            linewidth=1.1,
            color="0.25",
            zorder=3,
            label="state box",
        )
    )
    reference = entry["reference"]
    ax.plot(
        reference[0],
        reference[1],
        marker="*",
        markersize=15,
        color="tab:red",
        linestyle="none",
        zorder=5,
        label="reference point",
    )

    if zoom:
        span_x, span_y = model_bounds
    else:
        span_x = max(model_bounds[0], np.max(np.abs(vertices[:, 0])))
        span_y = max(model_bounds[1], np.max(np.abs(vertices[:, 1])))
    ax.set_xlim(-1.06 * span_x, 1.06 * span_x)
    ax.set_ylim(-1.06 * span_y, 1.06 * span_y)
    ax.set_xlabel(r"$x_{0,1}$")
    ax.set_ylabel(r"$x_{0,2}$")
    ax.grid(alpha=0.25)
    if entry["degenerate"]:
        headline = (
            r"$\mathcal{X}_{\mathrm{lin}}$ collapses to a segment "
            "(all $|u_k|$ equal)"
        )
    else:
        headline = (
            f"{100.0 * entry['valid_fraction']:.0f}% of "
            r"$\mathcal{X}_{\mathrm{lin}}$ feasible, covers "
            f"{100.0 * entry['covered_fraction']:.0f}% of "
            r"$\mathcal{X}$"
        )
    ax.set_title(
        f"$\\bar{{x}}_0=({reference[0]:.2f},\\,{reference[1]:.2f})$\n{headline}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)


def save_linearized_sets(entries: List[dict], output_dir: Path) -> None:
    """Store the overlap statistics and the polygon of every linearized set."""
    if not entries:
        return
    with (output_dir / "linearized_sets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "x0_1",
                "x0_2",
                "area_true",
                "area_linearized",
                "area_shared",
                "valid_fraction",
                "covered_fraction",
                "degenerate",
            ]
        )
        for index, entry in enumerate(entries):
            writer.writerow(
                [
                    index,
                    entry["reference"][0],
                    entry["reference"][1],
                    entry["area_true"],
                    entry["area_linearized"],
                    entry["area_shared"],
                    entry["valid_fraction"],
                    entry["covered_fraction"],
                    int(entry["degenerate"]),
                ]
            )

    rows = [
        np.column_stack(
            (np.full(entry["vertices"].shape[0], index), entry["vertices"])
        )
        for index, entry in enumerate(entries)
    ]
    np.savetxt(
        output_dir / "linearized_set_vertices.csv",
        np.vstack(rows),
        delimiter=",",
        header="index,x0_1,x0_2",
        comments="",
    )


def plot_statistics(results: List[SamplingResult], output_dir: Path) -> None:
    labels = [LABELS[result.method] for result in results]
    colors = [COLORS[result.method] for result in results]
    positions = np.arange(len(results))
    fig, axes = plt.subplots(1, 5, figsize=(23.5, 4.9))

    generation = np.array([result.generation_s for result in results])
    post_processing = np.array([result.post_processing_s for result in results])
    axes[0].bar(positions, generation, color=colors, label="generation")
    axes[0].bar(
        positions,
        post_processing,
        bottom=generation,
        color=colors,
        alpha=0.45,
        hatch="//",
        label="projection / rollout",
    )
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("Computation time")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].bar(positions, [result.seconds_per_sample for result in results], color=colors)
    axes[1].set_ylabel("seconds / accepted sample")
    axes[1].set_title("Cost per feasible state")

    # Solver calls are the hardware-independent counterpart of the wall times.
    per_sample = [result.solver_calls_per_sample for result in results]
    per_attempt = [result.solver_calls_per_attempt for result in results]
    width = 0.38
    axes[4].bar(
        positions - 0.5 * width, per_sample, width=width, color=colors,
        label="per accepted sample",
    )
    axes[4].bar(
        positions + 0.5 * width, per_attempt, width=width, color=colors,
        alpha=0.45, hatch="//", label="per attempt",
    )
    axes[4].set_ylabel("NLP solves")
    axes[4].set_title("Solver calls")
    axes[4].legend(loc="upper right", fontsize=8)

    scale = np.array(
        [100.0 / result.attempts if result.attempts else 0.0 for result in results]
    )
    infeasible = scale * np.array([result.infeasible_rejections for result in results])
    failures = scale * np.array([result.solver_failures for result in results])
    not_optimal = scale * np.array([result.optimality_rejections for result in results])
    axes[2].bar(positions, infeasible, color="0.6", label="infeasible candidate")
    axes[2].bar(positions, failures, bottom=infeasible, color="tab:purple", label="solver failure")
    axes[2].bar(
        positions,
        not_optimal,
        bottom=infeasible + failures,
        color="tab:brown",
        label="SOSC rejection",
    )
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_ylabel("rejection rate [%]")
    axes[2].set_title("Candidate rejection by cause")
    axes[2].legend(loc="upper right", fontsize=8)

    share = np.array(
        [100.0 / result.accepted if result.accepted else 0.0 for result in results]
    )
    analytic = share * np.array([result.analytic_feasible for result in results])
    optimal = share * np.array([result.locally_optimal for result in results])
    width = 0.38
    axes[3].bar(
        positions - 0.5 * width, analytic, width=width, color=colors, label="in feasible set"
    )
    axes[3].bar(
        positions + 0.5 * width,
        optimal,
        width=width,
        color=colors,
        alpha=0.45,
        hatch="//",
        label="SOSC satisfied",
    )
    # Both bars usually sit at 100%, so leave headroom for the legend.
    axes[3].set_ylim(0.0, 128.0)
    axes[3].set_yticks(np.arange(0.0, 101.0, 20.0))
    axes[3].set_ylabel("accepted samples [%]")
    axes[3].set_title("Feasibility and local optimality")
    axes[3].legend(loc="upper center", fontsize=8, ncol=2)

    for ax in axes:
        ax.set_xticks(positions, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Sampling strategy performance")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_performance.png", dpi=190)
    fig.savefig(output_dir / "sampling_performance.pdf", bbox_inches="tight")


def print_summary(results: List[SamplingResult]) -> None:
    print("\nSampling summary")
    print("-" * 146)
    print(
        f"{'method':<36} {'accepted':>9} {'in set':>7} {'SOSC':>6} {'attempts':>9} "
        f"{'solves':>8} {'/attempt':>9} {'infeas':>7} {'failed':>7} {'not opt':>8} "
        f"{'time [s]':>10} {'s/sample':>10}"
    )
    for result in results:
        print(
            f"{LABELS[result.method]:<36} {result.accepted:>9d} "
            f"{result.analytic_feasible:>7d} {result.locally_optimal:>6d} "
            f"{result.attempts:>9d} {result.solver_calls:>8d} "
            f"{result.solver_calls_per_attempt:>9.2f} "
            f"{result.infeasible_rejections:>7d} "
            f"{result.solver_failures:>7d} {result.optimality_rejections:>8d} "
            f"{result.elapsed_s:>10.3f} {result.seconds_per_sample:>10.4f}"
        )
    print(
        "\ncolumns: 'in set' = inside the analytic feasible set, 'SOSC' = accepted "
        "samples whose MPC\nsolution is a strict local minimizer. An attempt is one "
        "try at a feasible and optimal sample;\n'solves' counts the NLP solves those "
        "tries actually cost. Rejections are split into infeasible\ncandidates, "
        "failed solves and solutions that failed the local-optimality screen."
    )


def collect_all(
    model: DiscontinuousControl,
    args: argparse.Namespace,
    bounds: np.ndarray,
    model_bounds: np.ndarray,
    seed: int,
) -> List[SamplingResult]:
    """Run all four samplers once, with ``seed`` as their common random seed.

    Used both for the primary run (whose results also get verified, plotted
    and saved) and for the timing/acceptance-rate repeats, so a repeat is
    guaranteed to exercise exactly the same code path as the reported run.
    """
    return [
        collect_uniform(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=seed,
            model_bounds=model_bounds,
        ),
        collect_bilevel_projection(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=seed,
            projection_weight=args.projection_weight,
            verbose=args.verbose,
        ),
        collect_slacked_projection(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=seed,
            projection_weight=args.projection_weight,
            slack_penalty=args.slack_penalty,
            close_to_bounds=args.close_to_bounds,
            verbose=args.verbose,
        ),
        collect_bilevel_rollout(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            model_bounds=model_bounds,
            seed=seed,
            rollout_steps=args.rollout_steps,
            rollout_seeds=args.rollout_seeds,
            verbose=args.verbose,
            rollout_tol=args.rollout_tol,
            dedupe=args.rollout_dedupe,
        ),
    ]


def repeat_statistics(
    model: DiscontinuousControl,
    args: argparse.Namespace,
    bounds: np.ndarray,
    model_bounds: np.ndarray,
    primary_results: List[SamplingResult],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate timing and acceptance rate over ``args.repeats`` reruns.

    The primary run (already computed) is the first data point; feasibility
    and SOSC are not re-verified here, since they vary far less across seeds
    than timing does and the check itself is the expensive part of a repeat.
    """
    per_method: Dict[str, Dict[str, List[float]]] = {
        result.method: {
            "seconds_per_sample": [result.seconds_per_sample],
            "acceptance_rate": [result.acceptance_rate],
        }
        for result in primary_results
    }

    for repeat in range(1, args.repeats):
        for result in collect_all(model, args, bounds, model_bounds, args.seed + repeat):
            per_method[result.method]["seconds_per_sample"].append(result.seconds_per_sample)
            per_method[result.method]["acceptance_rate"].append(result.acceptance_rate)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, metrics in per_method.items():
        summary[method] = {}
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=float)
            summary[method][metric] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
                "values": [float(v) for v in array],
            }
    return summary


def print_paper_table(
    results: List[SamplingResult],
    stats: Dict[str, Dict[str, Dict[str, float]]],
    requested: int,
    repeats: int,
) -> None:
    """Print the condensed table to carry into a paper.

    Accepted count, feasibility and local optimality come from the primary
    run; timing and acceptance rate are mean +/- std over ``repeats`` runs
    (a single value, with std shown as "n/a", when ``repeats`` is 1).
    """
    print("\nCondensed results (for a paper)")
    print("-" * 100)
    header = (
        f"{'method':<36} {'accepted':>10} {'feasible':>9} {'SOSC':>7} "
        f"{'accept. rate [%]':>18} {'time/sample [ms]':>18}"
    )
    print(header)
    for result in results:
        accept = stats[result.method]["acceptance_rate"]
        timing = stats[result.method]["seconds_per_sample"]
        feasible_pct = (
            100.0 * result.analytic_feasible / result.accepted if result.accepted else float("nan")
        )
        sosc_pct = (
            100.0 * result.locally_optimal / result.accepted if result.accepted else float("nan")
        )
        if repeats > 1:
            accept_str = f"{100.0 * accept['mean']:.1f} +/- {100.0 * accept['std']:.1f}"
            time_str = f"{1000.0 * timing['mean']:.2f} +/- {1000.0 * timing['std']:.2f}"
        else:
            accept_str = f"{100.0 * accept['mean']:.1f}"
            time_str = f"{1000.0 * timing['mean']:.2f}"
        print(
            f"{LABELS[result.method]:<36} {result.accepted:>5d}/{requested:<4d} "
            f"{feasible_pct:>8.1f} {sosc_pct:>6.1f} {accept_str:>18} {time_str:>18}"
        )
    print("-" * 100)
    if repeats > 1:
        print(f"acceptance rate and time/sample are mean +/- std over {repeats} independent runs.")
    else:
        print(
            "acceptance rate and time/sample are from a single run; pass --repeats N "
            "for a mean +/- std suitable for a paper."
        )
    print(
        "'feasible' / 'SOSC' are the share of accepted samples inside the analytic "
        "feasible set / satisfying the second-order sufficient conditions (from the "
        "primary run only)."
    )


def save_paper_table(
    results: List[SamplingResult],
    stats: Dict[str, Dict[str, Dict[str, float]]],
    requested: int,
    repeats: int,
    output_dir: Path,
) -> None:
    with (output_dir / "paper_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "label",
                "requested",
                "accepted",
                "feasible_pct",
                "sosc_pct",
                "acceptance_rate_mean_pct",
                "acceptance_rate_std_pct",
                "seconds_per_sample_mean_ms",
                "seconds_per_sample_std_ms",
                "repeats",
            ]
        )
        for result in results:
            accept = stats[result.method]["acceptance_rate"]
            timing = stats[result.method]["seconds_per_sample"]
            feasible_pct = (
                100.0 * result.analytic_feasible / result.accepted if result.accepted else float("nan")
            )
            sosc_pct = (
                100.0 * result.locally_optimal / result.accepted if result.accepted else float("nan")
            )
            writer.writerow(
                [
                    result.method,
                    LABELS[result.method],
                    requested,
                    result.accepted,
                    feasible_pct,
                    sosc_pct,
                    100.0 * accept["mean"],
                    100.0 * accept["std"],
                    1000.0 * timing["mean"],
                    1000.0 * timing["std"],
                    repeats,
                ]
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.full(2, args.state_bound, dtype=float)
    model = DiscontinuousControl(N=args.horizon, sym_type="SX")
    model_bounds = np.asarray(model.state_bound, dtype=float)
    global MPC_RESTARTS
    MPC_RESTARTS = args.mpc_restarts
    profile = feasible_set_profile(model.N, model.control_bound)

    # Warm up the common MPC solver and the projection NLP outside all
    # measurements, so no method pays for one-time CasADi code generation.
    model.is_feasible(np.zeros(2))
    model.project_onto_feasible_set(np.zeros(2), Qx=args.projection_weight)

    results = collect_all(model, args, bounds, model_bounds, args.seed)
    verify_results(model, results, model_bounds)
    rows = [statistics_row(result, args.num_samples) for result in results]
    save_results(results, rows, args)
    save_feasible_set(profile, model_bounds, args.output_dir)
    plot_samples(
        results, bounds, model_bounds, model.control_bound, profile, args.output_dir
    )
    plot_post_processing(results, bounds, model_bounds, profile, args.output_dir)
    plot_statistics(results, args.output_dir)

    references = select_reference_points(results, args.linearization_points)
    entries = linearization_report(model, references, profile, model_bounds)
    save_linearized_sets(entries, args.output_dir)
    plot_linearized_sets(entries, profile, model_bounds, args.output_dir)
    for entry in entries:
        location = (
            f"X_lin at x0=({entry['reference'][0]:+.3f}, "
            f"{entry['reference'][1]:+.3f}): "
        )
        if entry["degenerate"]:
            print(
                location
                + "degenerate, it collapses to a segment because the |u_k| are "
                "all (nearly) equal there"
            )
        else:
            print(
                location
                + f"{100.0 * entry['valid_fraction']:.1f}% of it is truly feasible, "
                f"covering {100.0 * entry['covered_fraction']:.1f}% of X "
                f"(areas: {entry['area_linearized']:.2f} vs {entry['area_true']:.2f})"
            )
    print_summary(results)

    stats = repeat_statistics(model, args, bounds, model_bounds, results)
    save_paper_table(results, stats, args.num_samples, args.repeats, args.output_dir)
    print_paper_table(results, stats, args.num_samples, args.repeats)

    for result in results:
        if result.accepted < args.num_samples:
            print(
                f"note: {LABELS[result.method]} returned {result.accepted} of the "
                f"{args.num_samples} requested samples; it ran out of attempts "
                f"(--max-attempts {args.max_attempts})."
            )
    print(f"\nSaved samples, statistics, and plots to {args.output_dir.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
