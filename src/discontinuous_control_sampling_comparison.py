"""Compare initial-state sampling strategies for the discontinuous-control MPC.

The experiment compares uniform rejection sampling, the bilevel active-set
sampler, and locally linearized Hit-and-Run followed by nonlinear projection.
For every method it saves accepted states and first MPC controls, wall-clock
time, attempts, rejected candidates, and rejection rate.

Run from the repository root, for example::

    python3 src/discontinuous_control_sampling_comparison.py --num-samples 100

Results are written to ``results/discontinuous_control_sampling`` by default.
The bilevel method requires the CasADi ``ccopt`` plugin used by the model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import List, Tuple

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.discontinuous_control import DiscontinuousControl


LABELS = {
    "random": "Uniform random + rejection",
    "bilevel": "Bilevel active set",
    "hit_and_run": "Linearized Hit-and-Run",
}
COLORS = {
    "random": "tab:green",
    "bilevel": "tab:red",
    "hit_and_run": "tab:purple",
}


@dataclass
class SamplingResult:
    method: str
    samples: np.ndarray
    controls: np.ndarray
    elapsed_s: float
    attempts: int
    rejected: int
    seed_attempts: int = 0
    feasible_seeds: int = 0
    chain_updates: int = 0
    boundary_lp_failures: int = 0
    projection_needed: int = 0
    projection_failures: int = 0
    verified_feasible: int = 0

    @property
    def accepted(self) -> int:
        return int(self.samples.shape[0])

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempts if self.attempts else 0.0

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_s / self.accepted if self.accepted else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state-bound",
        type=float,
        default=4.0,
        help="Symmetric sampling bound for both initial-state components.",
    )
    parser.add_argument("--hr-num-seeds", type=int, default=5)
    parser.add_argument(
        "--hr-projection-weight",
        type=float,
        default=100.0,
        help="State-distance weight in the nonlinear Hit-and-Run projection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep solver/sampler progress output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "discontinuous_control_sampling",
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
    if args.hr_num_seeds <= 0:
        parser.error("--hr-num-seeds must be positive")
    if args.hr_projection_weight < 0.0:
        parser.error("--hr-projection-weight must be non-negative")
    return args


def normalize_samples(samples: object) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected samples with shape (n, 2), got {values.shape}.")
    return values


def solve_candidate(
    model: DiscontinuousControl, candidate: np.ndarray
) -> Tuple[bool, float]:
    """Solve the original MPC once and return feasibility and optimal first input."""
    try:
        solution = model.solver(
            x0=model.w0,
            lbx=model.lbw,
            ubx=model.ubw,
            lbg=model.lbg,
            ubg=model.ubg,
            p=candidate,
        )
    except Exception:
        return False, float("nan")
    if not model.solver.stats().get("success", False):
        return False, float("nan")

    solved_x0, controls = model.extract_traj(solution["x"])
    solved_x0 = solved_x0.full().reshape(2)
    controls = controls.full().reshape(model.N)
    terminal_state = solved_x0 + np.array(
        [np.sum(controls), np.sum(controls**3)]
    )
    feasible = bool(
        np.linalg.norm(solved_x0 - candidate, ord=np.inf) <= 1e-6
        and np.linalg.norm(terminal_state, ord=np.inf) <= 1e-6
        and np.all(np.abs(controls) <= model.control_bound + 1e-6)
    )
    return feasible, float(controls[0]) if feasible else float("nan")


def collect_random(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
) -> SamplingResult:
    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []
    controls: List[float] = []
    attempts = 0

    start = perf_counter()
    for attempts in range(1, max_attempts + 1):
        candidate = rng.uniform(-bounds, bounds)
        feasible, control = solve_candidate(model, candidate)
        if feasible:
            samples.append(candidate)
            controls.append(control)
        if len(samples) >= num_samples:
            break
    elapsed = perf_counter() - start

    values = normalize_samples(samples)
    return SamplingResult(
        method="random",
        samples=values,
        controls=np.asarray(controls, dtype=float),
        elapsed_s=elapsed,
        attempts=attempts,
        rejected=attempts - values.shape[0],
    )


def collect_bilevel(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    verbose: bool,
) -> SamplingResult:
    np.random.seed(seed)
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    start = perf_counter()
    try:
        with output:
            samples = model.collect_feasible_samples_with_bilevel(
                x_max=bounds,
                num_samples=num_samples,
                max_attempts=max_attempts,
                plot_results=False,
            )
    except Exception as exc:
        raise RuntimeError("The bilevel sampler failed. ") from exc
    elapsed = perf_counter() - start

    values = normalize_samples(samples)
    controls = np.asarray(model.last_feasible_sample_controls, dtype=float).reshape(-1)
    stats = model.last_feasible_sampling_stats
    attempts = int(stats["attempts"])
    return SamplingResult(
        method="bilevel",
        samples=values,
        controls=controls,
        elapsed_s=elapsed,
        attempts=attempts,
        rejected=attempts - values.shape[0],
    )


def collect_hit_and_run(
    model: DiscontinuousControl,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    num_seeds: int,
    projection_weight: float,
    verbose: bool,
) -> SamplingResult:
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    start = perf_counter()
    with output:
        samples = model.collect_feasible_samples_with_hit_and_run(
            num_samples=num_samples,
            num_feasible_seeds=num_seeds,
            max_seed_attempts=max_attempts,
            projection_Qx=projection_weight,
            random_seed=seed,
            x_max=bounds,
            plot_results=False,
            show=False,
        )
    elapsed = perf_counter() - start

    values = normalize_samples(samples)
    controls = np.asarray(model.last_feasible_sample_controls, dtype=float).reshape(-1)
    stats = model.last_linearized_sampling_stats
    raw_candidates = int(np.asarray(stats["raw_candidates"]).shape[0])
    projection_needed = int(stats["projection_needed"])
    projection_failures = int(stats["projection_failures"])
    return SamplingResult(
        method="hit_and_run",
        samples=values,
        controls=controls,
        elapsed_s=elapsed,
        # A rejection is a raw linearized proposal that is infeasible for the
        # original nonlinear MPC and consequently needs projection.
        attempts=raw_candidates,
        rejected=projection_needed,
        seed_attempts=int(stats["seed_attempts"]),
        feasible_seeds=int(np.asarray(stats["feasible_seeds"]).shape[0]),
        chain_updates=int(stats["chain_updates"]),
        boundary_lp_failures=int(stats["boundary_lp_failures"]),
        projection_needed=projection_needed,
        projection_failures=projection_failures,
    )


def verify_results(model: DiscontinuousControl, results: List[SamplingResult]) -> None:
    """Verify accepted states after timing, so checks do not bias wall times."""
    for result in results:
        result.verified_feasible = sum(model.is_feasible(sample) for sample in result.samples)


def statistics_row(result: SamplingResult, requested: int) -> dict:
    samples = result.samples
    radii = np.linalg.norm(samples, axis=1) if samples.size else np.empty(0)
    return {
        "method": result.method,
        "label": LABELS[result.method],
        "requested": requested,
        "accepted": result.accepted,
        "verified_feasible": result.verified_feasible,
        "attempts": result.attempts,
        "rejected": result.rejected,
        "rejection_rate": result.rejection_rate,
        "computation_time_s": result.elapsed_s,
        "seconds_per_accepted_sample": result.seconds_per_sample,
        "seed_attempts": result.seed_attempts,
        "feasible_seeds": result.feasible_seeds,
        "chain_updates": result.chain_updates,
        "boundary_lp_failures": result.boundary_lp_failures,
        "projection_needed": result.projection_needed,
        "projection_failures": result.projection_failures,
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
    header = "x0_1,x0_2,u0_opt"
    for result in results:
        data = np.column_stack((result.samples, result.controls))
        np.savetxt(
            output_dir / f"samples_{result.method}.csv",
            data,
            delimiter=",",
            header=header,
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
            "hr_num_seeds": args.hr_num_seeds,
            "hr_projection_weight": args.hr_projection_weight,
        },
        "rejection_rate_definition": (
            "For random and bilevel sampling: rejected candidates / attempted "
            "candidates. For Hit-and-Run: raw linearized candidates that are "
            "infeasible for the original nonlinear MPC and therefore need "
            "projection / all raw linearized candidates. Projection failures, "
            "seed attempts, and boundary-LP failures are reported separately."
        ),
        "optimality_notes": {
            "random": "Accepted after solving the original MPC with IPOPT.",
            "bilevel": "Accepted by the model's KKT/reduced-Hessian optimality screen.",
            "hit_and_run": "Accepted after the model's nonlinear feasible-set projection.",
        },
        "statistics": rows,
    }
    with (output_dir / "sampling_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def plot_samples(
    results: List[SamplingResult],
    bounds: np.ndarray,
    control_bound: float,
    output_dir: Path,
) -> None:
    if control_bound <= 0.0:
        raise ValueError("control_bound must be positive.")

    # Reserve a dedicated axis for the shared colorbar. Letting fig.colorbar
    # steal space from all three data axes and then calling subplots_adjust
    # produces a displaced or overlapping colorbar in some Matplotlib versions.
    fig = plt.figure(figsize=(15.5, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.12,
    )
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
        ]
    )
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    axes[2].sharex(axes[0])
    axes[2].sharey(axes[0])
    colorbar_axis = fig.add_subplot(grid[0, 3])

    # Use the physical input limits, rather than the extrema of one particular
    # sample batch, so a given color always denotes the same control value.
    normalization = matplotlib.colors.Normalize(
        vmin=-control_bound,
        vmax=control_bound,
    )
    colormap = plt.get_cmap("coolwarm")
    for ax, result in zip(axes, results):
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
            )
        ax.set_title(f"{LABELS[result.method]}\n$n={result.accepted}$")
        ax.set_xlim(-1.04 * bounds[0], 1.04 * bounds[0])
        ax.set_ylim(-1.04 * bounds[1], 1.04 * bounds[1])
        ax.set_xlabel(r"$x_{0,1}$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$x_{0,2}$")

    colorbar_mappable = matplotlib.cm.ScalarMappable(
        norm=normalization,
        cmap=colormap,
    )
    colorbar_mappable.set_array([])
    fig.colorbar(
        colorbar_mappable,
        cax=colorbar_axis,
        label=r"first optimal control $u_0^\star$",
    )
    # fig.suptitle("Feasible initial states for the discontinuous-control MPC")
    fig.savefig(output_dir / "sampled_initial_states.png", dpi=190)
    fig.savefig(output_dir / "sampled_initial_states.pdf", bbox_inches="tight")


def plot_statistics(
    results: List[SamplingResult], output_dir: Path
) -> None:
    labels = [LABELS[result.method] for result in results]
    colors = [COLORS[result.method] for result in results]
    positions = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    values = (
        [result.elapsed_s for result in results],
        [result.seconds_per_sample for result in results],
        [100.0 * result.rejection_rate for result in results],
    )
    ylabels = ("wall time [s]", "seconds / accepted sample", "rejection rate [%]")
    titles = ("Computation time", "Cost per feasible state", "Candidate rejection")
    for ax, data, ylabel, title in zip(axes, values, ylabels, titles):
        ax.bar(positions, data, color=colors)
        ax.set_xticks(positions, labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[2].set_ylim(0.0, 100.0)
    fig.suptitle("Sampling strategy performance")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_performance.png", dpi=190)
    fig.savefig(output_dir / "sampling_performance.pdf", bbox_inches="tight")


def print_summary(results: List[SamplingResult]) -> None:
    print("\nSampling summary")
    print("-" * 88)
    print(
        f"{'method':<28} {'accepted':>9} {'attempts':>9} "
        f"{'reject %':>10} {'time [s]':>11} {'s/sample':>11}"
    )
    for result in results:
        print(
            f"{LABELS[result.method]:<28} {result.accepted:>9d} "
            f"{result.attempts:>9d} {100.0 * result.rejection_rate:>10.2f} "
            f"{result.elapsed_s:>11.3f} {result.seconds_per_sample:>11.4f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = np.full(2, args.state_bound, dtype=float)
    model = DiscontinuousControl(N=args.horizon, sym_type="SX")

    # Warm up the common MPC solver outside all measurements.
    model.is_feasible(np.zeros(2))

    results = [
        collect_random(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
        ),
        collect_bilevel(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
            verbose=args.verbose,
        ),
        collect_hit_and_run(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
            num_seeds=args.hr_num_seeds,
            projection_weight=args.hr_projection_weight,
            verbose=args.verbose,
        ),
    ]
    verify_results(model, results)
    rows = [statistics_row(result, args.num_samples) for result in results]
    save_results(results, rows, args)
    plot_samples(results, bounds, model.control_bound, args.output_dir)
    plot_statistics(results, args.output_dir)
    print_summary(results)
    print(f"\nSaved samples, statistics, and plots to {args.output_dir.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
