"""Compare cart-pole feasible-state sampling strategies.

The experiment compares:

1. ``CartPole.collect_feasible_samples_with_bilevel``.
2. Uniform candidates followed by an MPC feasibility check.
3. Local linearized-MPC Hit-and-Run followed by nonlinear projection.
4. Bilevel-seeded Hit-and-Run followed by nonlinear projection (enabled by
   default and configurable with ``--hybrid-bilevel-hit-and-run``).

The quadratic- and L1-slack variants of
``CartPole.collect_feasible_samples`` can optionally be included with
``--slack-samplers``.

Run from the package root, for example::

    python src/cart_pole_feasible_sampling_comparison.py --num-samples 100

The script saves raw samples, a CSV summary, pairwise state projections,
state marginals, candidate feasibility diagnostics, and timing/coverage plots.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, List, Optional, Union

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.cart_pole import CartPole


STATE_NAMES = ("position", "velocity", "angle", "angular velocity")
STATE_UNITS = ("m", "m/s", "rad", "rad/s")
METHOD_LABELS = {
    "quadratic": "Quadratic slacks",
    "l1": "L1 slacks",
    "bilevel": "Bilevel active set",
    "uniform": "Uniform + MPC check",
    "linearized_hr": "Linearized MPC Hit-and-Run",
    "hybrid_bilevel_hr": "Bilevel-seeded Hit-and-Run",
}
METHOD_COLORS = {
    "quadratic": "tab:blue",
    "l1": "tab:orange",
    "bilevel": "tab:red",
    "uniform": "tab:green",
    "linearized_hr": "tab:purple",
    "hybrid_bilevel_hr": "tab:brown",
}


@dataclass
class SamplingResult:
    method: str
    samples: np.ndarray
    elapsed_s: float
    attempts: int
    successful_generation: int
    candidates: Optional[np.ndarray] = None
    candidate_feasible: Optional[np.ndarray] = None
    seed_attempts: int = 0
    feasible_seeds: int = 0
    candidate_attempts: int = 0
    projection_attempts: int = 0
    projection_needed: int = 0
    projection_failures: int = 0
    boundary_lp_failures: int = 0
    chain_updates: int = 0

    @property
    def accepted(self) -> int:
        return int(self.samples.shape[0])

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_s / self.accepted if self.accepted else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hybrid-bilevel-hit-and-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include Hit-and-Run initialized with bilevel-generated feasible "
            "seeds (enabled by default)."
        ),
    )
    parser.add_argument(
        "--slack-samplers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the quadratic- and L1-slack sampling strategies "
            "(disabled by default)."
        ),
    )
    parser.add_argument("--hr-num-seeds", type=int, default=5)
    parser.add_argument(
        "--hr-projection-weight",
        type=float,
        default=100.0,
        help="State-distance weight in the nonlinear Hit-and-Run projection.",
    )
    parser.add_argument(
        "--bounds-scale",
        type=float,
        default=1.0,
        help="Scale the initial-state sampling box for all enabled methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "cart_pole_feasible_sampling",
    )
    parser.add_argument(
        "--verbose-collector",
        action="store_true",
        help="Show per-attempt output from the model sampling methods.",
    )
    parser.add_argument("--show", action="store_true", help="Display plots after saving them.")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.hr_num_seeds <= 0:
        parser.error("--hr-num-seeds must be positive")
    if args.hr_projection_weight < 0.0:
        parser.error("--hr-projection-weight must be non-negative")
    if args.bounds_scale <= 0.0:
        parser.error("--bounds-scale must be positive")
    return args


def state_bounds(model: CartPole, scale: float) -> np.ndarray:
    return scale * np.array(
        [model.p_bound, model.v_bound, model.theta_bound, model.omega_bound],
        dtype=float,
    )


def normalize_samples(samples: object) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return np.empty((0, 4), dtype=float)
    if values.ndim == 2 and values.shape[0] == 4 and values.shape[1] != 4:
        values = values.T
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected samples with shape (n, 4), got {values.shape}.")
    return values


def collect_optimized(
    model: CartPole,
    penalty: str,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    verbose: bool,
) -> SamplingResult:
    # Resetting the legacy global RNG gives both penalties identical random
    # residual-weight sequences, making their comparison reproducible.
    np.random.seed(seed)
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    start = perf_counter()
    with output:
        samples = model.collect_feasible_samples_with_slacked(
            x_max=bounds,
            close_to_bounds=True,
            num_samples=num_samples,
            max_attempts=max_attempts,
            plot_results=False,
            slack_penalty=penalty,
        )
    elapsed = perf_counter() - start
    samples = normalize_samples(samples)
    stats = getattr(model, "last_feasible_sampling_stats", {})
    attempts = int(stats.get("attempts", max_attempts))
    return SamplingResult(
        method=penalty,
        samples=samples,
        elapsed_s=elapsed,
        attempts=attempts,
        successful_generation=int(stats.get("successful_extreme_solves", samples.shape[0])),
        candidate_attempts=attempts,
    )


def collect_uniform(
    model: CartPole,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
) -> SamplingResult:
    rng = np.random.default_rng(seed)
    feasible_samples: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    candidate_feasible: List[bool] = []

    start = perf_counter()
    for _ in range(max_attempts):
        candidate = rng.uniform(-bounds, bounds)
        feasible = model.is_feasible(candidate)
        candidates.append(candidate)
        candidate_feasible.append(feasible)
        if feasible:
            feasible_samples.append(candidate)
        if len(feasible_samples) >= num_samples:
            break
    elapsed = perf_counter() - start

    return SamplingResult(
        method="uniform",
        samples=normalize_samples(feasible_samples),
        elapsed_s=elapsed,
        attempts=len(candidates),
        successful_generation=len(candidates),
        candidates=normalize_samples(candidates),
        candidate_feasible=np.asarray(candidate_feasible, dtype=bool),
        candidate_attempts=len(candidates),
    )


def collect_bilevel(
    model: CartPole,
    num_samples: int,
    max_attempts: int,
    bounds: np.ndarray,
    seed: int,
    verbose: bool,
) -> SamplingResult:
    np.random.seed(seed)
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    start = perf_counter()
    with output:
        samples = model.collect_feasible_samples_with_bilevel(
            x_max=bounds,
            num_samples=num_samples,
            max_attempts=max_attempts,
            plot_results=False,
        )
    elapsed = perf_counter() - start
    samples = normalize_samples(samples)
    stats = getattr(model, "last_feasible_sampling_stats", {})
    attempts = int(stats.get("attempts", max_attempts))
    return SamplingResult(
        method="bilevel",
        samples=samples,
        elapsed_s=elapsed,
        attempts=attempts,
        successful_generation=int(
            stats.get("feasible_and_optimal_candidates", samples.shape[0])
        ),
        candidate_attempts=attempts,
    )


def collect_linearized_hit_and_run(
    model: CartPole,
    num_samples: int,
    max_seed_attempts: int,
    bounds: np.ndarray,
    seed: int,
    num_feasible_seeds: int,
    projection_weight: float,
    verbose: bool,
    use_bilevel_seeds: bool = False,
) -> SamplingResult:
    model_bounds = np.array(
        [model.p_bound, model.v_bound, model.theta_bound, model.omega_bound],
        dtype=float,
    )
    # The nonlinear MPC itself cannot admit initial states beyond its bounds.
    hr_bounds = np.minimum(bounds, model_bounds)
    if use_bilevel_seeds:
        # The bilevel residual-weight sampler uses NumPy's global RNG.
        np.random.seed(seed)
    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    start = perf_counter()
    with output:
        samples = model.collect_linearized_hit_and_run_samples(
            num_samples=num_samples,
            num_feasible_seeds=num_feasible_seeds,
            max_seed_attempts=max_seed_attempts,
            projection_Qx=projection_weight,
            use_bilevel=use_bilevel_seeds,
            random_seed=seed,
            x_max=hr_bounds,
            plot_results=False,
            show=False,
        )
    elapsed = perf_counter() - start
    stats = model.last_linearized_sampling_stats
    raw_candidates = normalize_samples(stats["raw_candidates"])
    seed_attempts = int(stats["seed_attempts"])
    candidate_attempts = int(raw_candidates.shape[0])
    projection_needed = int(stats["projection_needed"])
    projection_failures = int(stats["projection_failures"])
    # The latest implementation projects every candidate and then determines
    # whether that projection changed the state.
    projection_attempts = candidate_attempts

    return SamplingResult(
        method="hybrid_bilevel_hr" if use_bilevel_seeds else "linearized_hr",
        samples=normalize_samples(samples),
        elapsed_s=elapsed,
        # Use raw chain proposals as the denominator, consistent with the
        # sampler's current rejection/projection statistics. Seed searches and
        # failed boundary LPs remain visible in their dedicated columns.
        attempts=candidate_attempts,
        successful_generation=int(stats["chain_updates"]),
        candidates=raw_candidates,
        seed_attempts=seed_attempts,
        feasible_seeds=int(np.asarray(stats["feasible_seeds"]).shape[0]),
        candidate_attempts=candidate_attempts,
        projection_attempts=projection_attempts,
        projection_needed=projection_needed,
        projection_failures=projection_failures,
        boundary_lp_failures=int(stats["boundary_lp_failures"]),
        chain_updates=int(stats["chain_updates"]),
    )


SummaryValue = Union[float, int, str]


def coverage_metrics(result: SamplingResult, bounds: np.ndarray) -> Dict[str, SummaryValue]:
    samples = result.samples
    normalized = samples / bounds if samples.size else np.empty((0, 4))
    boundary_score = np.max(np.abs(normalized), axis=1) if samples.size else np.array([])

    occupied_bins = 0
    if samples.size:
        # A compact four-dimensional coverage score: number of occupied cells
        # in a 4 x 4 x 4 x 4 grid over the common state box.
        bin_indices = np.clip(((normalized + 1.0) * 2.0).astype(int), 0, 3)
        occupied_bins = int(np.unique(bin_indices, axis=0).shape[0])

    row: Dict[str, SummaryValue] = {
        "method": result.method,
        "requested": int(result.accepted if result.accepted else 0),
        "accepted": result.accepted,
        "attempts": result.attempts,
        "successful_generation": result.successful_generation,
        "seed_attempts": result.seed_attempts,
        "feasible_seeds": result.feasible_seeds,
        "candidate_attempts": result.candidate_attempts,
        "projection_attempts": result.projection_attempts,
        "projection_needed": result.projection_needed,
        "projection_failures": result.projection_failures,
        "boundary_lp_failures": result.boundary_lp_failures,
        "chain_updates": result.chain_updates,
        "acceptance_rate": result.acceptance_rate,
        "elapsed_s": result.elapsed_s,
        "seconds_per_accepted_sample": result.seconds_per_sample,
        "unique_samples": int(np.unique(samples, axis=0).shape[0]) if samples.size else 0,
        "occupied_4d_bins_of_256": occupied_bins,
        "mean_boundary_score": float(np.mean(boundary_score)) if boundary_score.size else float("nan"),
        "fraction_boundary_score_ge_0_9": (
            float(np.mean(boundary_score >= 0.9)) if boundary_score.size else float("nan")
        ),
    }
    for idx, name in enumerate(("position", "velocity", "angle", "angular_velocity")):
        row[f"normalized_{name}_mean"] = (
            float(np.mean(normalized[:, idx])) if normalized.size else float("nan")
        )
        row[f"normalized_{name}_std"] = (
            float(np.std(normalized[:, idx])) if normalized.size else float("nan")
        )
        row[f"normalized_{name}_min"] = (
            float(np.min(normalized[:, idx])) if normalized.size else float("nan")
        )
        row[f"normalized_{name}_max"] = (
            float(np.max(normalized[:, idx])) if normalized.size else float("nan")
        )
    return row


def save_summary(rows: List[Dict[str, SummaryValue]], output_dir: Path) -> None:
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_samples(results: List[SamplingResult], output_dir: Path) -> None:
    header = "position,velocity,angle,angular_velocity"
    for result in results:
        np.savetxt(
            output_dir / f"samples_{result.method}.csv",
            result.samples,
            delimiter=",",
            header=header,
            comments="",
        )

    uniform = next(result for result in results if result.method == "uniform")
    uniform_data = np.column_stack((uniform.candidates, uniform.candidate_feasible.astype(int)))
    np.savetxt(
        output_dir / "uniform_candidates.csv",
        uniform_data,
        delimiter=",",
        header=f"{header},mpc_feasible",
        comments="",
    )

    linearized_hr = next(result for result in results if result.method == "linearized_hr")
    np.savetxt(
        output_dir / "linearized_hr_candidates.csv",
        linearized_hr.candidates,
        delimiter=",",
        header=header,
        comments="",
    )

    hybrid = next(
        (result for result in results if result.method == "hybrid_bilevel_hr"),
        None,
    )
    if hybrid is not None:
        np.savetxt(
            output_dir / "hybrid_bilevel_hr_candidates.csv",
            hybrid.candidates,
            delimiter=",",
            header=header,
            comments="",
        )


def plot_pairwise(results: List[SamplingResult], bounds: np.ndarray, output_dir: Path) -> None:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (i, j) in zip(axes.ravel(), pairs):
        for result in results:
            if result.samples.size:
                ax.scatter(
                    result.samples[:, i],
                    result.samples[:, j],
                    s=22,
                    alpha=0.52,
                    color=METHOD_COLORS[result.method],
                    edgecolors="none",
                    label=METHOD_LABELS[result.method],
                )
        ax.set_xlim(-1.04 * bounds[i], 1.04 * bounds[i])
        ax.set_ylim(-1.04 * bounds[j], 1.04 * bounds[j])
        ax.set_xlabel(f"{STATE_NAMES[i]} [{STATE_UNITS[i]}]")
        ax.set_ylabel(f"{STATE_NAMES[j]} [{STATE_UNITS[j]}]")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Feasible cart-pole samples: all pairwise state projections")
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_feasible_samples.png", dpi=180)
    fig.savefig(output_dir / "pairwise_feasible_samples.pdf", bbox_inches="tight")


def plot_marginals(results: List[SamplingResult], bounds: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for state_idx, ax in enumerate(axes.ravel()):
        bins = np.linspace(-bounds[state_idx], bounds[state_idx], 25)
        for result in results:
            if result.samples.size:
                ax.hist(
                    result.samples[:, state_idx],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    color=METHOD_COLORS[result.method],
                    label=METHOD_LABELS[result.method],
                )
        ax.set_xlabel(f"{STATE_NAMES[state_idx]} [{STATE_UNITS[state_idx]}]")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Marginal distributions of accepted feasible samples")
    fig.tight_layout()
    fig.savefig(output_dir / "state_marginals.png", dpi=180)
    fig.savefig(output_dir / "state_marginals.pdf", bbox_inches="tight")


def plot_uniform_checks(uniform: SamplingResult, bounds: np.ndarray, output_dir: Path) -> None:
    pairs = ((0, 2), (2, 3), (0, 1), (1, 3))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    mask = uniform.candidate_feasible
    candidates = uniform.candidates
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.scatter(
            candidates[~mask, i],
            candidates[~mask, j],
            s=18,
            alpha=0.35,
            color="tab:red",
            label="MPC infeasible",
        )
        ax.scatter(
            candidates[mask, i],
            candidates[mask, j],
            s=22,
            alpha=0.65,
            color="tab:green",
            label="MPC feasible",
        )
        ax.set_xlim(-1.04 * bounds[i], 1.04 * bounds[i])
        ax.set_ylim(-1.04 * bounds[j], 1.04 * bounds[j])
        ax.set_xlabel(f"{STATE_NAMES[i]} [{STATE_UNITS[i]}]")
        ax.set_ylabel(f"{STATE_NAMES[j]} [{STATE_UNITS[j]}]")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Uniform candidates after MPC feasibility checks")
    fig.tight_layout()
    fig.savefig(output_dir / "uniform_feasibility_checks.png", dpi=180)
    fig.savefig(output_dir / "uniform_feasibility_checks.pdf", bbox_inches="tight")


def plot_performance(
    results: List[SamplingResult],
    rows: List[Dict[str, SummaryValue]],
    output_dir: Path,
) -> None:
    labels = [METHOD_LABELS[result.method] for result in results]
    colors = [METHOD_COLORS[result.method] for result in results]
    x_pos = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(x_pos, [result.elapsed_s for result in results], color=colors)
    axes[0].set_ylabel("total wall time [s]")
    axes[0].set_title("Sampling time")

    axes[1].bar(x_pos, [result.seconds_per_sample for result in results], color=colors)
    axes[1].set_ylabel("seconds / accepted sample")
    axes[1].set_title("Cost per feasible sample")

    axes[2].bar(x_pos, [float(row["acceptance_rate"]) for row in rows], color=colors)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("accepted / attempted")
    axes[2].set_title("Acceptance rate")

    for ax in axes:
        ax.set_xticks(x_pos, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Computational comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_performance.png", dpi=180)
    fig.savefig(output_dir / "sampling_performance.pdf", bbox_inches="tight")


def print_summary(
    results: List[SamplingResult], rows: List[Dict[str, SummaryValue]]
) -> None:
    print("\nSampling summary")
    print("-" * 92)
    print(
        f"{'method':<24} {'accepted':>10} {'attempts':>10} "
        f"{'rate':>10} {'time [s]':>12} {'s/sample':>12}"
    )
    for result, row in zip(results, rows):
        print(
            f"{METHOD_LABELS[result.method]:<24} "
            f"{result.accepted:>10d} {result.attempts:>10d} "
            f"{float(row['acceptance_rate']):>10.3f} "
            f"{result.elapsed_s:>12.3f} {result.seconds_per_sample:>12.4f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = CartPole(args.horizon, dt=args.dt, sym_type="SX")
    bounds = state_bounds(model, args.bounds_scale)

    # Exclude one-time MPC solver initialization from all measured methods.
    model.is_feasible(np.zeros(4))

    results = []
    if args.slack_samplers:
        results.extend([
            collect_optimized(
                model,
                penalty="quadratic",
                num_samples=args.num_samples,
                max_attempts=args.max_attempts,
                bounds=bounds,
                seed=args.seed,
                verbose=args.verbose_collector,
            ),
            collect_optimized(
                model,
                penalty="l1",
                num_samples=args.num_samples,
                max_attempts=args.max_attempts,
                bounds=bounds,
                seed=args.seed,
                verbose=args.verbose_collector,
            ),
        ])

    results.extend([
        collect_bilevel(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
            verbose=args.verbose_collector,
        ),
        collect_uniform(
            model,
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
        ),
        collect_linearized_hit_and_run(
            model,
            num_samples=args.num_samples,
            max_seed_attempts=args.max_attempts,
            bounds=bounds,
            seed=args.seed,
            num_feasible_seeds=args.hr_num_seeds,
            projection_weight=args.hr_projection_weight,
            verbose=args.verbose_collector,
        ),
    ])

    if args.hybrid_bilevel_hit_and_run:
        results.append(
            collect_linearized_hit_and_run(
                model,
                num_samples=args.num_samples,
                max_seed_attempts=args.max_attempts,
                bounds=bounds,
                seed=args.seed,
                num_feasible_seeds=args.hr_num_seeds,
                projection_weight=args.hr_projection_weight,
                verbose=args.verbose_collector,
                use_bilevel_seeds=True,
            )
        )

    rows = [coverage_metrics(result, bounds) for result in results]
    for row in rows:
        row["requested"] = args.num_samples

    save_summary(rows, args.output_dir)
    save_samples(results, args.output_dir)
    plot_pairwise(results, bounds, args.output_dir)
    plot_marginals(results, bounds, args.output_dir)
    uniform_result = next(result for result in results if result.method == "uniform")
    plot_uniform_checks(uniform_result, bounds, args.output_dir)
    plot_performance(results, rows, args.output_dir)
    print_summary(results, rows)
    print(f"\nSaved results to {args.output_dir.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
