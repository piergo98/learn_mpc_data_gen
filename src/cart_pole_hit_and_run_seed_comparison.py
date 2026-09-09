"""Compare cart-pole Hit-and-Run with uniform and bilevel seed generation.

Both runs use the same Hit-and-Run and nonlinear-projection settings. The only
difference is how the feasible seeds are generated.

Run from the repository root, for example::

    python src/cart_pole_hit_and_run_seed_comparison.py --num-samples 100
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
from typing import Dict, List

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
STRATEGIES = ("uniform", "bilevel")
LABELS = {
    "uniform": "Hit-and-Run (uniform seeds)",
    "bilevel": "Hit-and-Run (bilevel seeds)",
}
COLORS = {"uniform": "tab:purple", "bilevel": "tab:brown"}


@dataclass
class HitAndRunResult:
    strategy: str
    samples: np.ndarray
    proposals: np.ndarray
    projected_mask: np.ndarray
    seeds: np.ndarray
    controls: np.ndarray
    elapsed_s: float
    stats: Dict[str, object]

    @property
    def accepted(self) -> int:
        return int(self.samples.shape[0])

    @property
    def candidate_attempts(self) -> int:
        return int(np.asarray(self.stats["raw_candidates"]).shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--projection-weight", type=float, default=100.0)
    parser.add_argument("--bounds-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "cart_pole_hit_and_run_seed_comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep solver and sampler progress output.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if args.projection_weight < 0.0:
        parser.error("--projection-weight must be non-negative")
    if args.bounds_scale <= 0.0:
        parser.error("--bounds-scale must be positive")
    return args


def normalize_states(values: object) -> np.ndarray:
    states = np.asarray(values, dtype=float)
    if states.size == 0:
        return np.empty((0, 4), dtype=float)
    if states.ndim == 2 and states.shape[0] == 4 and states.shape[1] != 4:
        states = states.T
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"Expected states with shape (n, 4), got {states.shape}.")
    return states


def collect(
    model: CartPole,
    strategy: str,
    args: argparse.Namespace,
    bounds: np.ndarray,
) -> HitAndRunResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown seed strategy: {strategy}")

    # Bilevel residual weights use the legacy global RNG; Hit-and-Run itself
    # uses ``random_seed``. Reset both runs for a reproducible comparison.
    np.random.seed(args.seed)
    output = (
        contextlib.nullcontext()
        if args.verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    start = perf_counter()
    with output:
        samples = model.collect_linearized_hit_and_run_samples(
            num_samples=args.num_samples,
            num_feasible_seeds=args.num_seeds,
            max_seed_attempts=args.max_attempts,
            projection_Qx=args.projection_weight,
            use_bilevel=strategy == "bilevel",
            random_seed=args.seed,
            x_max=bounds,
            plot_results=False,
            show=False,
        )
    elapsed_s = perf_counter() - start

    stats = dict(model.last_linearized_sampling_stats)
    samples = normalize_states(samples)
    proposals = normalize_states(stats["accepted_raw_candidates"])
    projected_mask = np.asarray(stats["accepted_projected"], dtype=bool).reshape(-1)
    seeds = normalize_states(stats["feasible_seeds"])
    controls = np.asarray(model.last_feasible_sample_controls, dtype=float).reshape(-1)
    if not (
        proposals.shape[0]
        == projected_mask.size
        == controls.size
        == samples.shape[0]
    ):
        raise RuntimeError(
            "Hit-and-Run diagnostics are not aligned with the returned samples."
        )

    return HitAndRunResult(
        strategy=strategy,
        samples=samples,
        proposals=proposals,
        projected_mask=projected_mask,
        seeds=seeds,
        controls=controls,
        elapsed_s=elapsed_s,
        stats=stats,
    )


def summary_row(result: HitAndRunResult, bounds: np.ndarray) -> Dict[str, object]:
    normalized = result.samples / bounds
    boundary_score = np.max(np.abs(normalized), axis=1)
    candidate_attempts = result.candidate_attempts
    return {
        "seed_strategy": result.strategy,
        "requested": int(result.stats["requested_samples"]),
        "accepted": result.accepted,
        "candidate_attempts": candidate_attempts,
        "acceptance_rate": (
            result.accepted / candidate_attempts if candidate_attempts else 0.0
        ),
        "elapsed_s": result.elapsed_s,
        "seconds_per_sample": result.elapsed_s / result.accepted,
        "seed_attempts": int(result.stats["seed_attempts"]),
        "feasible_seeds": int(result.seeds.shape[0]),
        "chain_updates": int(result.stats["chain_updates"]),
        "boundary_lp_failures": int(result.stats["boundary_lp_failures"]),
        "projection_needed": int(result.stats["projection_needed"]),
        "projection_failures": int(result.stats["projection_failures"]),
        "projected_samples": int(np.count_nonzero(result.projected_mask)),
        "projection_fraction": float(np.mean(result.projected_mask)),
        "unique_samples": int(np.unique(result.samples, axis=0).shape[0]),
        "mean_boundary_score": float(np.mean(boundary_score)),
    }


def save_results(
    results: List[HitAndRunResult],
    rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    for result in results:
        np.savez_compressed(
            output_dir / f"hit_and_run_{result.strategy}.npz",
            projected_samples=result.samples,
            linearized_proposals=result.proposals,
            projected_mask=result.projected_mask,
            feasible_seeds=result.seeds,
            first_controls=result.controls,
        )

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pairwise(
    results: List[HitAndRunResult],
    bounds: np.ndarray,
    output_dir: Path,
) -> None:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (state_i, state_j) in zip(axes.ravel(), pairs):
        for result in results:
            ax.scatter(
                result.samples[:, state_i],
                result.samples[:, state_j],
                s=30,
                alpha=0.58,
                color=COLORS[result.strategy],
                edgecolors="none",
                label=LABELS[result.strategy],
            )
        ax.set_xlim(-1.05 * bounds[state_i], 1.05 * bounds[state_i])
        ax.set_ylim(-1.05 * bounds[state_j], 1.05 * bounds[state_j])
        ax.set_xlabel(f"{STATE_NAMES[state_i]} [{STATE_UNITS[state_i]}]")
        ax.set_ylabel(f"{STATE_NAMES[state_j]} [{STATE_UNITS[state_j]}]")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", framealpha=0.95)
    fig.suptitle("Projected Hit-and-Run samples by seed-generation strategy")
    fig.savefig(output_dir / "projected_samples_pairwise.png", dpi=190)
    fig.savefig(
        output_dir / "projected_samples_pairwise.pdf", bbox_inches="tight"
    )


def plot_3d(
    results: List[HitAndRunResult],
    bounds: np.ndarray,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for result in results:
        ax.scatter(
            result.samples[:, 0],
            result.samples[:, 1],
            result.samples[:, 2],
            s=38,
            alpha=0.68,
            color=COLORS[result.strategy],
            edgecolors="none",
            label=LABELS[result.strategy],
            depthshade=False,
        )
    ax.set_xlim(-1.05 * bounds[0], 1.05 * bounds[0])
    ax.set_ylim(-1.05 * bounds[1], 1.05 * bounds[1])
    ax.set_zlim(-1.05 * bounds[2], 1.05 * bounds[2])
    ax.set_xlabel("Position [m]", labelpad=9)
    ax.set_ylabel("Velocity [m/s]", labelpad=9)
    ax.set_zlabel("Angle [rad]", labelpad=9)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_title("Projected Hit-and-Run samples: position, velocity, and angle")
    fig.savefig(output_dir / "projected_samples_3d.png", dpi=190)
    fig.savefig(output_dir / "projected_samples_3d.pdf", bbox_inches="tight")


def plot_performance(
    results: List[HitAndRunResult],
    rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    labels = [LABELS[result.strategy] for result in results]
    colors = [COLORS[result.strategy] for result in results]
    positions = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].bar(positions, [result.elapsed_s for result in results], color=colors)
    axes[0].set_title("Wall time")
    axes[0].set_ylabel("seconds")
    axes[1].bar(
        positions,
        [float(row["acceptance_rate"]) for row in rows],
        color=colors,
    )
    axes[1].set_title("Proposal acceptance")
    axes[1].set_ylabel("accepted / raw proposals")
    axes[1].set_ylim(0.0, 1.05)
    axes[2].bar(
        positions,
        [float(row["projection_fraction"]) for row in rows],
        color=colors,
    )
    axes[2].set_title("Projection fraction")
    axes[2].set_ylabel("projected / accepted")
    axes[2].set_ylim(0.0, 1.05)
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "performance_comparison.png", dpi=190)
    fig.savefig(output_dir / "performance_comparison.pdf", bbox_inches="tight")


def save_diagnostic_plots(
    model: CartPole,
    results: List[HitAndRunResult],
    bounds: np.ndarray,
    output_dir: Path,
) -> None:
    for result in results:
        for view_3d, suffix in ((False, "pairwise"), (True, "3d")):
            fig, _ = model.plot_linearized_hit_and_run_samples(
                result.samples,
                linearization_samples=result.proposals,
                projected_mask=result.projected_mask,
                seeds=result.seeds,
                x_max=bounds,
                view_3d=view_3d,
                show=False,
            )
            fig.savefig(
                output_dir / f"{result.strategy}_seed_diagnostics_{suffix}.png",
                dpi=190,
                bbox_inches="tight",
            )


def print_summary(
    results: List[HitAndRunResult], rows: List[Dict[str, object]]
) -> None:
    print("\nHit-and-Run seed comparison")
    print("-" * 88)
    print(
        f"{'seed strategy':<30} {'accepted':>10} {'proposals':>10} "
        f"{'time [s]':>12} {'projected':>12}"
    )
    for result, row in zip(results, rows):
        print(
            f"{LABELS[result.strategy]:<30} "
            f"{result.accepted:>10d} {result.candidate_attempts:>10d} "
            f"{result.elapsed_s:>12.3f} "
            f"{int(row['projected_samples']):>12d}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = CartPole(args.horizon, dt=args.dt, sym_type="SX")
    model_bounds = np.array(
        [model.p_bound, model.v_bound, model.theta_bound, model.omega_bound],
        dtype=float,
    )
    bounds = np.minimum(args.bounds_scale * model_bounds, model_bounds)

    # Exclude one-time MPC initialization from both measured methods.
    model.is_feasible(np.zeros(4))
    results = [
        collect(model, strategy, args, bounds)
        for strategy in STRATEGIES
    ]
    rows = [summary_row(result, bounds) for result in results]

    save_results(results, rows, args.output_dir)
    plot_pairwise(results, bounds, args.output_dir)
    plot_3d(results, bounds, args.output_dir)
    plot_performance(results, rows, args.output_dir)
    save_diagnostic_plots(model, results, bounds, args.output_dir)
    print_summary(results, rows)
    print(f"\nSaved results to {args.output_dir.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
