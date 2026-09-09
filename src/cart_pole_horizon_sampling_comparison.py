"""Cart-pole sampling comparison across MPC horizons.

Runs the same four initial-state samplers compared for the discontinuous-control
model and reports the same condensed table, once per horizon, so the effect of
the horizon ``N`` on data generation is visible:

1. ``uniform``            -- uniform candidates in the state box, kept only when
   the MPC is feasible for them.
2. ``bilevel_projection`` -- candidates from the bilevel active-set sampler,
   each one afterwards mapped through the nonlinear feasible-set projection.
3. ``slacked_projection`` -- candidates from the slacked extreme-state sampler,
   each one afterwards mapped through the same projection.
4. ``bilevel_rollout``    -- bilevel seeds propagated with closed-loop MPC
   rollouts, so every visited state becomes a sample.

Run from the repository root, for example::

    python3 src/cart_pole_horizon_sampling_comparison.py --horizons 5 10 15 20 \\
        --num-samples 50 --repeats 3

Results go to ``results/cart_pole_horizon_sampling`` by default. The bilevel
methods require the CasADi ``ccopt`` plugin used by the model.

Differences from the discontinuous-control study
------------------------------------------------
* The cart pole has no closed-form feasible set, so the "feasible" column is
  **solver-verified**: an MPC solve that converges, reproduces the requested
  ``x0`` and satisfies every bound. There is no analytic cross-check available
  to catch a solver false negative, unlike in the 2-D study.
* The MPC bounds its whole predicted trajectory, not only ``x_0``, so a
  closed-loop successor cannot leave the state box the way it does there.

Notes
-----
* Timing and acceptance rate are averaged over ``--repeats`` independent runs
  per horizon; feasibility and the SOSC check come from the first run only,
  because they vary far less across seeds and the check is the expensive part.
* All verification happens after the timers stop, so it never biases the
  reported wall times.
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

from models.cart_pole import CartPole


STATE_DIM = 4
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
MARKERS = {
    "uniform": "o",
    "bilevel_projection": "s",
    "slacked_projection": "^",
    "bilevel_rollout": "D",
}
# A projection displacement below this norm means the candidate was already on
# the feasible set, so the projection only re-solved the MPC.
PROJECTION_MOVE_TOL = 1e-8
# Extra MPC initial guesses tried before a state is reported infeasible.
MPC_RESTARTS = 1
# A rollout stops once the closed loop has essentially reached the origin.
ROLLOUT_ORIGIN_TOL = 1e-6
# Bound on the seed batches drawn to reach the requested sample count.
ROLLOUT_MAX_BATCHES = 8
# Safety factor on a top-up batch: the marginal yield keeps falling as states
# accumulate, so asking for exactly the projected count lands short.
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
    locally_optimal: int = 0
    sosc_degenerate: int = 0
    mpc_restarts: int = 0
    solver_calls: int = 0
    verification_solver_calls: int = 0
    infeasible_rejections: int = 0
    solver_failures: int = 0
    optimality_rejections: int = 0

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
        """Accepted samples per attempt, where an attempt is one try at a
        feasible and optimal sample."""
        return self.accepted / self.attempts if self.attempts else float("nan")

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_s / self.accepted if self.accepted else float("nan")

    @property
    def solver_calls_per_attempt(self) -> float:
        return self.solver_calls / self.attempts if self.attempts else float("nan")

    @property
    def solver_calls_per_sample(self) -> float:
        return self.solver_calls / self.accepted if self.accepted else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="MPC horizons to sweep. One condensed table is printed per horizon.",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Independent reruns per horizon used to report timing and "
            "acceptance rate as mean +/- std. The first run also supplies the "
            "feasibility and SOSC columns, which are not repeated."
        ),
    )
    parser.add_argument(
        "--bounds-scale",
        type=float,
        default=1.0,
        help="Scale the initial-state sampling box relative to the MPC state box.",
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
            "0 tops up batches until --num-samples states exist."
        ),
    )
    parser.add_argument(
        "--rollout-dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop a rollout when it revisits a state already collected.",
    )
    parser.add_argument(
        "--rollout-tol",
        type=float,
        default=ROLLOUT_ORIGIN_TOL,
        help="Stop a closed-loop rollout once the state is this close to the origin.",
    )
    parser.add_argument(
        "--mpc-restarts",
        type=int,
        default=MPC_RESTARTS,
        help="Extra MPC initial guesses tried before a state is called infeasible.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep solver/sampler progress output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "cart_pole_horizon_sampling",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if any(horizon < 1 for horizon in args.horizons):
        parser.error("every horizon must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.bounds_scale <= 0.0:
        parser.error("--bounds-scale must be positive")
    if args.projection_weight < 0.0:
        parser.error("--projection-weight must be non-negative")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.rollout_seeds < 0:
        parser.error("--rollout-seeds must be non-negative")
    if args.rollout_tol < 0.0:
        parser.error("--rollout-tol must be non-negative")
    if args.mpc_restarts < 0:
        parser.error("--mpc-restarts must be non-negative")
    return args


def normalize_samples(samples: object) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.empty((0, STATE_DIM), dtype=float)
    if values.ndim == 2 and values.shape[0] == STATE_DIM and values.shape[1] != STATE_DIM:
        values = values.T
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError(f"Expected samples with shape (n, {STATE_DIM}), got {values.shape}.")
    return values


def quiet(verbose: bool):
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


class SolverCalls:
    """Counts NLP solves.

    An attempt is one try at producing a feasible and optimal sample; it is not
    the same as one solver call. A retried MPC solve costs several calls, and a
    projected candidate costs its generator solve plus the projection solve.
    """

    def __init__(self) -> None:
        self.count = 0

    def add(self, calls: int = 1) -> None:
        self.count += calls


def mpc_warm_starts(model: CartPole, x0: np.ndarray, restarts: int, rng: np.random.Generator):
    """Yield MPC initial guesses for ``x0``.

    The first is the model's own guess. The second coasts from ``x0`` under
    zero input, which respects the dynamics equalities and is a far better
    starting point than the origin for a state far from upright. Further
    guesses perturb that coast trajectory, so a failed solve gets a genuinely
    different starting point rather than the same one twice.
    """
    yield np.asarray(model.w0, dtype=float)

    horizon = model.N
    states = np.empty((STATE_DIM, horizon + 1))
    states[:, 0] = x0
    for k in range(horizon):
        states[:, k + 1] = np.asarray(
            model.mpc_step(states[:, k], 0.0).full(), dtype=float
        ).reshape(STATE_DIM)
    inputs = np.zeros((1, horizon))
    coast = np.asarray(model.traj_to_vec(states, inputs).full(), dtype=float).reshape(-1)
    yield coast

    for _ in range(max(restarts - 1, 0)):
        noise = rng.normal(scale=0.1, size=coast.shape)
        yield np.clip(coast + noise, model.lbw, model.ubw)


def solve_mpc(
    model: CartPole,
    x0: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    restarts: Optional[int] = None,
    tol: float = 1e-6,
    calls: Optional[SolverCalls] = None,
) -> Optional[dict]:
    """Solve the MPC at ``x0``, retrying from fresh guesses.

    Returns the primal solution, the predicted trajectory, the equality and
    bound multipliers and the number of extra restarts needed, or ``None`` when
    no guess produced a solution satisfying the constraints.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.shape != (STATE_DIM,) or not np.all(np.isfinite(x0)):
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
        states, inputs = model.extract_traj(solution["x"])
        states = states.full()
        inputs = inputs.full().reshape(-1)
        # The equality residual covers both the initial-state constraint and
        # the shooting dynamics, so it is the honest feasibility check here.
        equality_residual = np.linalg.norm(model.h(x0, w).full().reshape(-1), ord=np.inf)
        if (
            equality_residual > tol
            or np.any(np.abs(states) > model.x_max[:, None] + tol)
            or np.any(np.abs(inputs) > model.u_bound + tol)
        ):
            continue

        return {
            "w": w,
            "x0": states[:, 0],
            "states": states,
            "controls": inputs,
            "lam_h": solution["lam_g"].full().reshape(-1),
            "lam_x": solution["lam_x"].full().reshape(-1),
            "restarts": attempt,
        }
    return None


def solve_candidate(
    model: CartPole,
    candidate: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    calls: Optional[SolverCalls] = None,
) -> Tuple[bool, float]:
    """Return MPC feasibility of ``candidate`` and its optimal first input."""
    solution = solve_mpc(model, candidate, rng=rng, calls=calls)
    if solution is None:
        return False, float("nan")
    return True, float(solution["controls"][0])


def check_sosc(model: CartPole, solution: dict, tol: float = 1e-6) -> dict:
    """Check the second-order sufficient conditions at an MPC solution.

    ``Z`` spans the null space of the equality Jacobian stacked with the
    Jacobians of the active bound constraints, and SOSC holds when the reduced
    Hessian ``Z' H Z`` is positive definite. The bound constraints are linear,
    so their multipliers do not enter ``H`` and only the equality multipliers
    are needed.

    An active constraint whose multiplier vanishes makes ``Z`` a strict subset
    of the critical cone, which turns the test conservative; those are counted
    in ``degenerate`` so they stay visible.
    """
    w = solution["w"]
    residual = model.g(w).full().reshape(-1)
    active = residual >= -tol

    jac_h = model.jac_h(solution["x0"], w).full()
    jac_g = model.jac_g(w).full()
    active_jacobian = np.vstack([jac_h, jac_g[active, :]]) if np.any(active) else jac_h

    # Each bound row of g touches exactly one variable, so its multiplier is
    # the corresponding bound multiplier, signed by the row's direction.
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


def project_candidates(
    model: CartPole,
    candidates: np.ndarray,
    projection_weight: float,
    num_samples: int,
    calls: Optional[SolverCalls] = None,
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """Project candidates onto the nonlinear MPC feasible set.

    The upright state is the projection warm start. It is solved once here and
    handed to every projection, because ``project_onto_feasible_set``
    otherwise re-solves that same MPC on every call.
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
        p=np.zeros(STATE_DIM),
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
    model: CartPole,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
) -> SamplingResult:
    rng = np.random.default_rng(seed)
    solver_rng = np.random.default_rng(seed + 1)
    samples: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    accepted_mask: List[bool] = []
    calls = SolverCalls()

    start = perf_counter()
    for _ in range(max_attempts):
        candidate = rng.uniform(-bounds, bounds)
        candidates.append(candidate)
        feasible, _ = solve_candidate(model, candidate, rng=solver_rng, calls=calls)
        accepted_mask.append(feasible)
        if feasible:
            samples.append(candidate)
        if len(samples) >= num_samples:
            break
    elapsed = perf_counter() - start

    rejected = len(candidates) - len(samples)
    return SamplingResult(
        method="uniform",
        samples=normalize_samples(samples),
        elapsed_s=elapsed,
        generation_s=elapsed,
        attempts=len(candidates),
        candidates=normalize_samples(candidates),
        candidate_feasible=np.asarray(accepted_mask, dtype=bool),
        solver_calls=calls.count,
        # Without a closed-form feasible set these cannot be separated: a
        # rejected draw is reported as infeasible, which also absorbs any
        # solver false negative.
        infeasible_rejections=rejected,
    )


def collect_bilevel_projection(
    model: CartPole,
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
    model: CartPole,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    projection_weight: float,
    slack_penalty: str,
    close_to_bounds: bool,
    verbose: bool,
) -> SamplingResult:
    # The spiked residual weights also come from the legacy global RNG.
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
    # The slacked sampler has no optimality screen of its own: local optimality
    # is only measured afterwards by the SOSC check in verify_results.
    solved = int(stats["successful_extreme_solves"])
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
    model: CartPole,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    rollout_steps: int,
    rollout_seeds: int,
    verbose: bool,
    rollout_tol: float = ROLLOUT_ORIGIN_TOL,
    dedupe: bool = True,
) -> SamplingResult:
    """Propagate bilevel seeds with closed-loop MPC and keep every visited state.

    The successor is the MPC's own predicted next state, so it satisfies the
    state box by construction. A rollout stops when the state reaches the
    origin, when it repeats one already collected, or when a solve fails.
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
    converged = 0
    duplicates = 0
    batches = 0
    calls = SolverCalls()
    batch_yield: Optional[float] = None
    seen: set = set()
    remaining_attempts = max_attempts

    def register(state: np.ndarray) -> bool:
        """Record a state, reporting whether it is new."""
        if not dedupe:
            return True
        key = tuple(np.round(state, 9))
        if key in seen:
            return False
        seen.add(key)
        return True

    while len(samples) < num_samples and remaining_attempts > 0 and batches < ROLLOUT_MAX_BATCHES:
        missing = num_samples - len(samples)
        if explicit_seeds:
            requested = rollout_seeds
        elif batch_yield is not None:
            # Size the top-up from the marginal yield of the previous batch:
            # the cumulative average over-estimates what the next batch returns.
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
                solution = solve_mpc(model, state, rng=solver_rng, calls=calls)
                simulated_steps += 1
                if solution is None:
                    rollout_failures += 1
                    break
                # The MPC's own predicted successor: inside the state box by
                # construction, since the MPC bounds its whole trajectory.
                state = solution["states"][:, 1]
                if not register(state):
                    duplicates += 1
                    break
                trajectory.append(state)
                samples.append(state)
            trajectories.append(np.asarray(trajectory, dtype=float))
        rollout_s += perf_counter() - start

        new_seeds = len(seed_states) - batch_start_seeds
        if new_seeds > 0:
            batch_yield = max((len(samples) - batch_start_samples) / new_seeds, 0.1)

        if explicit_seeds:
            break

    return SamplingResult(
        method="bilevel_rollout",
        samples=normalize_samples(samples),
        solver_calls=calls.count,
        elapsed_s=generation_s + rollout_s,
        generation_s=generation_s,
        post_processing_s=rollout_s,
        attempts=attempts + simulated_steps,
        candidates=normalize_samples(seed_states),
        solver_failures=solver_failures + rollout_failures,
        optimality_rejections=optimality_rejections,
        rollout_seeds=len(seed_states),
        rollout_steps=rollout_steps,
        rollout_failures=rollout_failures,
        rollout_converged=converged,
        rollout_duplicates=duplicates,
        rollout_batches=batches,
        trajectories=trajectories,
    )


def verify_results(model: CartPole, results: List[SamplingResult]) -> None:
    """Recompute controls and check feasibility and local optimality.

    Runs after all timings, so neither the SOSC test nor the extra MPC solves
    bias the measured wall times.
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


def collect_all(
    model: CartPole,
    args: argparse.Namespace,
    bounds: np.ndarray,
    seed: int,
) -> List[SamplingResult]:
    """Run all four samplers once, with ``seed`` as their common random seed."""
    return [
        collect_uniform(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=seed,
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
            seed=seed,
            rollout_steps=args.rollout_steps,
            rollout_seeds=args.rollout_seeds,
            verbose=args.verbose,
            rollout_tol=args.rollout_tol,
            dedupe=args.rollout_dedupe,
        ),
    ]


def repeat_statistics(
    model: CartPole,
    args: argparse.Namespace,
    bounds: np.ndarray,
    primary_results: List[SamplingResult],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate timing and acceptance rate over ``args.repeats`` reruns."""
    per_method: Dict[str, Dict[str, List[float]]] = {
        result.method: {
            "seconds_per_sample": [result.seconds_per_sample],
            "acceptance_rate": [result.acceptance_rate],
        }
        for result in primary_results
    }

    for repeat in range(1, args.repeats):
        for result in collect_all(model, args, bounds, args.seed + repeat):
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


def print_summary(results: List[SamplingResult], horizon: int) -> None:
    print(f"\nSampling summary (N = {horizon})")
    print("-" * 132)
    print(
        f"{'method':<36} {'accepted':>9} {'MPC ok':>7} {'SOSC':>6} {'attempts':>9} "
        f"{'solves':>8} {'/attempt':>9} {'rejected':>9} {'time [s]':>10} {'s/sample':>10}"
    )
    for result in results:
        print(
            f"{LABELS[result.method]:<36} {result.accepted:>9d} "
            f"{result.verified_feasible:>7d} {result.locally_optimal:>6d} "
            f"{result.attempts:>9d} {result.solver_calls:>8d} "
            f"{result.solver_calls_per_attempt:>9.2f} {result.rejected:>9d} "
            f"{result.elapsed_s:>10.3f} {result.seconds_per_sample:>10.4f}"
        )


def paper_rows(
    results: List[SamplingResult],
    stats: Dict[str, Dict[str, Dict[str, float]]],
    horizon: int,
    requested: int,
    repeats: int,
) -> List[dict]:
    rows = []
    for result in results:
        accept = stats[result.method]["acceptance_rate"]
        timing = stats[result.method]["seconds_per_sample"]
        rows.append(
            {
                "horizon": horizon,
                "method": result.method,
                "label": LABELS[result.method],
                "requested": requested,
                "accepted": result.accepted,
                "feasible_pct": (
                    100.0 * result.verified_feasible / result.accepted
                    if result.accepted
                    else float("nan")
                ),
                "sosc_pct": (
                    100.0 * result.locally_optimal / result.accepted
                    if result.accepted
                    else float("nan")
                ),
                "acceptance_rate_mean_pct": 100.0 * accept["mean"],
                "acceptance_rate_std_pct": 100.0 * accept["std"],
                "seconds_per_sample_mean_ms": 1000.0 * timing["mean"],
                "seconds_per_sample_std_ms": 1000.0 * timing["std"],
                "solver_calls_per_sample": result.solver_calls_per_sample,
                "repeats": repeats,
            }
        )
    return rows


def print_paper_table(rows: List[dict], horizon: int, repeats: int) -> None:
    """Print the condensed table for one horizon."""
    print(f"\nCondensed results (for a paper) -- N = {horizon}")
    print("-" * 100)
    print(
        f"{'method':<36} {'accepted':>10} {'feasible':>9} {'SOSC':>7} "
        f"{'accept. rate [%]':>18} {'time/sample [ms]':>18}"
    )
    for row in rows:
        if repeats > 1:
            accept_str = (
                f"{row['acceptance_rate_mean_pct']:.1f} +/- "
                f"{row['acceptance_rate_std_pct']:.1f}"
            )
            time_str = (
                f"{row['seconds_per_sample_mean_ms']:.1f} +/- "
                f"{row['seconds_per_sample_std_ms']:.1f}"
            )
        else:
            accept_str = f"{row['acceptance_rate_mean_pct']:.1f}"
            time_str = f"{row['seconds_per_sample_mean_ms']:.1f}"
        print(
            f"{row['label']:<36} {row['accepted']:>5d}/{row['requested']:<4d} "
            f"{row['feasible_pct']:>8.1f} {row['sosc_pct']:>6.1f} "
            f"{accept_str:>18} {time_str:>18}"
        )
    print("-" * 100)


def print_horizon_sweep(rows: List[dict], horizons: List[int], repeats: int) -> None:
    """Print the whole sweep as one table, method-major, for easy comparison."""
    print("\nHorizon sweep")
    print("=" * 118)
    print(
        f"{'method':<36} {'N':>4} {'accepted':>10} {'feasible':>9} {'SOSC':>7} "
        f"{'accept. rate [%]':>18} {'time/sample [ms]':>18}"
    )
    for method in LABELS:
        for horizon in horizons:
            row = next(
                (r for r in rows if r["method"] == method and r["horizon"] == horizon),
                None,
            )
            if row is None:
                continue
            if repeats > 1:
                accept_str = (
                    f"{row['acceptance_rate_mean_pct']:.1f} +/- "
                    f"{row['acceptance_rate_std_pct']:.1f}"
                )
                time_str = (
                    f"{row['seconds_per_sample_mean_ms']:.1f} +/- "
                    f"{row['seconds_per_sample_std_ms']:.1f}"
                )
            else:
                accept_str = f"{row['acceptance_rate_mean_pct']:.1f}"
                time_str = f"{row['seconds_per_sample_mean_ms']:.1f}"
            print(
                f"{row['label']:<36} {horizon:>4d} "
                f"{row['accepted']:>5d}/{row['requested']:<4d} "
                f"{row['feasible_pct']:>8.1f} {row['sosc_pct']:>6.1f} "
                f"{accept_str:>18} {time_str:>18}"
            )
        print("-" * 118)
    if repeats > 1:
        print(
            f"acceptance rate and time/sample are mean +/- std over {repeats} "
            "independent runs per horizon."
        )
    else:
        print(
            "acceptance rate and time/sample are from a single run per horizon; "
            "pass --repeats N for mean +/- std."
        )
    print(
        "'feasible' is solver-verified (the cart pole has no closed-form feasible "
        "set); 'SOSC' is the share of accepted samples whose MPC solution is a "
        "strict local minimizer. Both come from the first run of each horizon."
    )


def save_sweep(rows: List[dict], args: argparse.Namespace) -> None:
    with (args.output_dir / "horizon_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "configuration": {
            "horizons": args.horizons,
            "num_samples": args.num_samples,
            "max_attempts": args.max_attempts,
            "dt": args.dt,
            "seed": args.seed,
            "repeats": args.repeats,
            "bounds_scale": args.bounds_scale,
            "projection_weight": args.projection_weight,
            "slack_penalty": args.slack_penalty,
            "close_to_bounds": args.close_to_bounds,
            "rollout_steps": args.rollout_steps,
            "rollout_seeds": args.rollout_seeds,
            "rollout_dedupe": args.rollout_dedupe,
            "rollout_tol": args.rollout_tol,
            "mpc_restarts": args.mpc_restarts,
        },
        "column_definitions": {
            "accepted": "samples returned for the requested budget",
            "feasible_pct": (
                "share of accepted samples for which the MPC solve converges, "
                "reproduces x0 and satisfies every bound. Solver-verified: the "
                "cart pole has no closed-form feasible set to cross-check."
            ),
            "sosc_pct": (
                "share of accepted samples whose MPC solution satisfies the "
                "second-order sufficient conditions (reduced Hessian positive "
                "definite on the null space of the active constraints)."
            ),
            "acceptance_rate": (
                "accepted / attempts, where an attempt is one try at a feasible "
                "and optimal sample. Mean +/- std over independent runs."
            ),
            "seconds_per_sample": "wall time per accepted sample, mean +/- std.",
        },
        "rows": rows,
    }
    with (args.output_dir / "horizon_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def plot_horizon_trends(rows: List[dict], horizons: List[int], output_dir: Path) -> None:
    """Plot how the horizon moves each reported quantity."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    panels = (
        ("seconds_per_sample_mean_ms", "seconds_per_sample_std_ms", "time / accepted sample [ms]", True),
        ("acceptance_rate_mean_pct", "acceptance_rate_std_pct", "acceptance rate [%]", False),
        ("feasible_pct", None, "MPC-feasible samples [%]", False),
        ("sosc_pct", None, "SOSC satisfied [%]", False),
    )
    for ax, (mean_key, std_key, ylabel, log_scale) in zip(axes, panels):
        for method, label in LABELS.items():
            xs, ys, errs = [], [], []
            for horizon in horizons:
                row = next(
                    (r for r in rows if r["method"] == method and r["horizon"] == horizon),
                    None,
                )
                if row is None:
                    continue
                xs.append(horizon)
                ys.append(row[mean_key])
                errs.append(row[std_key] if std_key else 0.0)
            if not xs:
                continue
            errs = np.nan_to_num(np.asarray(errs, dtype=float))
            ax.errorbar(
                xs,
                ys,
                yerr=errs if np.any(errs > 0) else None,
                marker=MARKERS[method],
                color=COLORS[method],
                capsize=3,
                linewidth=1.6,
                markersize=6,
                label=label,
            )
        ax.set_xlabel("horizon $N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.25)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.0, 105.0)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Effect of the MPC horizon on cart-pole data generation")
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_trends.png", dpi=190)
    fig.savefig(output_dir / "horizon_trends.pdf", bbox_inches="tight")


def save_samples(results: List[SamplingResult], horizon: int, output_dir: Path) -> None:
    header = "position,velocity,angle,angular_velocity,u0_opt"
    for result in results:
        controls = result.controls
        if controls.size != result.accepted:
            controls = np.full(result.accepted, np.nan)
        np.savetxt(
            output_dir / f"samples_N{horizon}_{result.method}.csv",
            np.column_stack((result.samples, controls)),
            delimiter=",",
            header=header,
            comments="",
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global MPC_RESTARTS
    MPC_RESTARTS = args.mpc_restarts

    all_rows: List[dict] = []
    for horizon in args.horizons:
        print(f"\n{'=' * 60}\nHorizon N = {horizon}\n{'=' * 60}")
        model = CartPole(horizon, dt=args.dt, sym_type="SX")
        bounds = args.bounds_scale * np.asarray(model.x_max, dtype=float)

        # Warm up the MPC solver and the projection NLP outside all
        # measurements, so no method pays for one-time CasADi code generation.
        model.is_feasible(np.zeros(STATE_DIM))
        model.project_onto_feasible_set(np.zeros(STATE_DIM), Qx=args.projection_weight)

        results = collect_all(model, args, bounds, args.seed)
        verify_results(model, results)
        stats = repeat_statistics(model, args, bounds, results)
        rows = paper_rows(results, stats, horizon, args.num_samples, args.repeats)
        all_rows.extend(rows)

        save_samples(results, horizon, args.output_dir)
        print_summary(results, horizon)
        print_paper_table(rows, horizon, args.repeats)
        for result in results:
            if result.accepted < args.num_samples:
                print(
                    f"note: {LABELS[result.method]} returned {result.accepted} of the "
                    f"{args.num_samples} requested samples; it ran out of attempts "
                    f"(--max-attempts {args.max_attempts})."
                )

    save_sweep(all_rows, args)
    plot_horizon_trends(all_rows, args.horizons, args.output_dir)
    print_horizon_sweep(all_rows, args.horizons, args.repeats)
    print(f"\nSaved samples, statistics, and plots to {args.output_dir.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
