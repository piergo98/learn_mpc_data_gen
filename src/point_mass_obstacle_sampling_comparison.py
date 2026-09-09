"""Compare three initial-state samplers on a nonconvex point-mass problem.

A planar double integrator must travel from A to B with zero velocity while
five dense walls of circular obstacles force it through alternating openings.
The experiment compares the three strategies demonstrated by ``point_mass``:
uniform rejection, randomly weighted residual slacks with a quadratic penalty,
and locally linearized Hit-and-Run.

Run from the ``ultro`` package directory, for example::

    python3 src/point_mass_obstacle_sampling_comparison.py --num-samples 12

The script writes raw trajectories, a CSV summary, and comparison plots to
``results/point_mass_obstacle_sampling`` by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, List

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.point_mass import PointMass


METHODS = ("uniform", "quadratic", "hit_and_run")
LABELS = {
    "uniform": "Uniform + rejection",
    "quadratic": "Quadratic slacks",
    "hit_and_run": "Linearized Hit-and-Run",
}
COLORS = {
    "uniform": "tab:green",
    "quadratic": "tab:blue",
    "hit_and_run": "tab:purple",
}


@dataclass
class SamplingResult:
    method: str
    coordinates: np.ndarray
    elapsed_s: float
    stats: Dict[str, object]
    requested: int

    @property
    def accepted(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def attempts(self) -> int:
        return int(self.stats.get("attempts", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hr-num-seeds", type=int, default=5)
    parser.add_argument("--hr-burn-in", type=int, default=10)
    parser.add_argument("--hr-thinning", type=int, default=1)
    parser.add_argument("--hr-lp-solver", default="qpoases")
    parser.add_argument(
        "--no-hr-projection",
        action="store_true",
        help="Discard nonlinear-infeasible Hit-and-Run proposals instead of projecting them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "point_mass_obstacle_sampling",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.max_attempts <= 0:
        parser.error("--num-samples and --max-attempts must be positive")
    if args.horizon < 6 or args.dt <= 0.0:
        parser.error("use --horizon >= 6 and --dt > 0")
    if args.hr_num_seeds <= 0:
        parser.error("--hr-num-seeds must be positive")
    if args.hr_burn_in < 0 or args.hr_thinning <= 0:
        parser.error("invalid Hit-and-Run burn-in or thinning")
    return args


def run_sampler(
    model: PointMass,
    method: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> SamplingResult:
    start = perf_counter()
    if method == "uniform":
        coordinates, _, uniform_stats = model.sample_uniform(
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            rng=rng,
        )
        stats = dict(uniform_stats)
    elif method == "quadratic":
        np.random.seed(int(rng.integers(2**31 - 1)))
        coordinates = model.collect_feasible_samples(
            num_samples=args.num_samples,
            max_attempts=args.max_attempts,
            plot_results=False,
            slack_penalty="quadratic",
        )
        stats = dict(model.last_feasible_sampling_stats)
    elif method == "hit_and_run":
        coordinates = model.collect_linearized_hit_and_run_samples(
            num_samples=args.num_samples,
            num_feasible_seeds=args.hr_num_seeds,
            max_seed_attempts=args.max_attempts,
            burn_in=args.hr_burn_in,
            thinning=args.hr_thinning,
            project_infeasible=not args.no_hr_projection,
            random_seed=int(rng.integers(2**31 - 1)),
            plot_results=False,
            show=False,
            lp_solver=args.hr_lp_solver,
        )
        hr_stats = model.last_linearized_sampling_stats
        stats = {
            "attempts": int(hr_stats["seed_attempts"] + hr_stats["chain_attempts"]),
            "seed_attempts": int(hr_stats["seed_attempts"]),
            "chain_attempts": int(hr_stats["chain_attempts"]),
            "chain_steps": int(hr_stats["chain_updates"]),
            "raw_candidates": int(hr_stats["raw_candidates"].shape[0]),
            "nonlinear_feasible_candidates": int(
                np.count_nonzero(hr_stats["raw_nonlinear_feasible"])
            ),
            "projected_candidates": int(np.count_nonzero(hr_stats["accepted_projected"])),
            "projection_failures": int(hr_stats["projection_failures"]),
            "boundary_lp_failures": int(hr_stats["boundary_lp_failures"]),
            "mpc_solve_failures": int(hr_stats["mpc_solve_failures"]),
        }
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return SamplingResult(method, coordinates, perf_counter() - start, stats, args.num_samples)


def trajectory_metrics(model: PointMass, coordinates: np.ndarray) -> Dict[str, float]:
    if coordinates.shape[0] == 0:
        return {
            "mean_path_length": float("nan"),
            "mean_control_energy": float("nan"),
            "mean_min_clearance": float("nan"),
            "occupied_position_bins": 0,
        }
    trajectories = [model.solve_MPC_trajectory(x0) for x0 in coordinates]
    paths = np.asarray([states[:, :2] for states, _ in trajectories])
    controls = np.asarray([controls for _, controls in trajectories])
    lengths = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)
    energies = model.dt * np.sum(controls**2, axis=(1, 2))
    clearances = []
    for states, controls in trajectories:
        positions = []
        for k in range(model.N):
            for substep in range(model.collision_substeps):
                tau = substep * model.dt_mpc / model.collision_substeps
                positions.append(
                    states[k, :2] + tau * states[k, 2:] + 0.5 * tau**2 * controls[k]
                )
        positions.append(states[-1, :2])
        positions = np.asarray(positions)
        distance_margins = []
        for obstacle in model.obstacles:
            distance_margins.append(
                np.linalg.norm(positions - np.asarray(obstacle.center), axis=1)
                - obstacle.radius
                - model.body_radius
            )
        clearances.append(np.min(distance_margins))

    xmin, xmax, ymin, ymax = model.workspace
    normalized = np.column_stack((
        (paths[:, :, 0].reshape(-1) - xmin) / (xmax - xmin),
        (paths[:, :, 1].reshape(-1) - ymin) / (ymax - ymin),
    ))
    bins = np.clip((normalized * 20).astype(int), 0, 19)
    return {
        "mean_path_length": float(np.mean(lengths)),
        "mean_control_energy": float(np.mean(energies)),
        "mean_min_clearance": float(np.mean(clearances)),
        "occupied_position_bins": int(np.unique(bins, axis=0).shape[0]),
    }


def save_data(
    model: PointMass,
    results: List[SamplingResult],
    output_dir: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        saved_coordinates: List[np.ndarray] = []
        trajectories = []
        for x0 in result.coordinates:
            candidate = np.asarray(x0, dtype=float).reshape(4)
            try:
                trajectory = model.solve_MPC_trajectory(candidate)
            except RuntimeError:
                try:
                    candidate = model.project_onto_feasible_set(candidate)
                    trajectory = model.solve_MPC_trajectory(candidate)
                except RuntimeError:
                    continue
            saved_coordinates.append(candidate)
            trajectories.append(trajectory)
        if trajectories:
            states = np.asarray([states for states, _ in trajectories])
            controls = np.asarray([controls for _, controls in trajectories])
        else:
            states = np.empty((0, model.horizon + 1, 4))
            controls = np.empty((0, model.horizon, 2))
        saved_coordinates_array = np.asarray(saved_coordinates, dtype=float).reshape(-1, 4)
        np.savez_compressed(
            output_dir / f"samples_{result.method}.npz",
            initial_states=saved_coordinates_array,
            states=states,
            controls=controls,
            dt=model.dt,
            start=model.start,
            goal=model.goal,
        )
        metrics = trajectory_metrics(model, saved_coordinates_array)
        row: Dict[str, object] = {
            "method": result.method,
            "requested": result.requested,
            "accepted": result.accepted,
            "attempts": result.attempts,
            "acceptance_rate": result.accepted / result.attempts if result.attempts else 0.0,
            "elapsed_s": result.elapsed_s,
            "seconds_per_sample": (
                result.elapsed_s / result.accepted if result.accepted else float("nan")
            ),
            **metrics,
            **{f"stat_{key}": value for key, value in result.stats.items() if key != "attempts"},
        }
        rows.append(row)

    fieldnames: List[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def draw_course(ax: plt.Axes, model: PointMass) -> None:
    for obstacle in model.obstacles:
        ax.add_patch(Circle(obstacle.center, obstacle.radius, color="0.25", alpha=0.75))
        ax.add_patch(
            Circle(
                obstacle.center,
                obstacle.radius + model.body_radius,
                facecolor="none",
                edgecolor="0.4",
                linestyle=":",
                linewidth=0.7,
            )
        )
    ax.scatter(*model.goal, marker="*", s=150, color="gold", edgecolor="black", zorder=5)
    ax.annotate("B", model.goal, xytext=(5, 5), textcoords="offset points", weight="bold")
    xmin, xmax, ymin, ymax = model.workspace
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x position", ylabel="y position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)


def plot_trajectories(
    model: PointMass,
    results: List[SamplingResult],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1, len(results), figsize=(5.2 * len(results), 5.2), sharex=True, sharey=True,
        squeeze=False,
    )
    for ax, result in zip(axes.ravel(), results):
        draw_course(ax, model)
        initial_positions = []
        for x0 in result.coordinates:
            candidate = np.asarray(x0, dtype=float).reshape(4)
            try:
                path, _ = model.solve_MPC_trajectory(candidate)
            except RuntimeError:
                try:
                    candidate = model.project_onto_feasible_set(candidate)
                    path, _ = model.solve_MPC_trajectory(candidate)
                except RuntimeError:
                    continue
            path = path[:, :2]
            initial_positions.append(path[0])
            ax.plot(path[:, 0], path[:, 1], color=COLORS[result.method], alpha=0.42, linewidth=1.25)
        if initial_positions:
            initial_positions = np.asarray(initial_positions)
            ax.scatter(
                initial_positions[:, 0],
                initial_positions[:, 1],
                marker="o",
                s=48,
                color=COLORS[result.method],
                edgecolor="black",
                linewidth=0.7,
                zorder=6,
                label="initial states",
            )
            ax.legend(loc="best")
        ax.set_title(f"{LABELS[result.method]} ({result.accepted} accepted)")
    fig.suptitle("Feasible double-integrator trajectories through a nonconvex obstacle field")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=190)
    fig.savefig(output_dir / "trajectory_comparison.pdf", bbox_inches="tight")


def plot_performance(
    results: List[SamplingResult], rows: List[Dict[str, object]], output_dir: Path
) -> None:
    labels = [LABELS[result.method] for result in results]
    colors = [COLORS[result.method] for result in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].bar(x, [result.elapsed_s for result in results], color=colors)
    axes[0].set_title("Wall time")
    axes[0].set_ylabel("seconds")
    axes[1].bar(x, [result.accepted for result in results], color=colors)
    axes[1].set_title("Accepted initial states")
    axes[1].set_ylabel("count")
    axes[2].bar(x, [float(row["occupied_position_bins"]) for row in rows], color=colors)
    axes[2].set_title("Spatial coverage")
    axes[2].set_ylabel("occupied 20 x 20 bins")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sampling_performance.png", dpi=190)
    fig.savefig(output_dir / "sampling_performance.pdf", bbox_inches="tight")


def print_summary(results: List[SamplingResult]) -> None:
    print("\nSampling summary")
    print("-" * 79)
    print(f"{'method':<20} {'accepted':>9} {'attempts':>10} {'rate':>10} {'time [s]':>12}")
    for result in results:
        rate = result.accepted / result.attempts if result.attempts else 0.0
        print(
            f"{LABELS[result.method]:<20} {result.accepted:>9d} {result.attempts:>10d} "
            f"{rate:>10.3f} {result.elapsed_s:>12.3f}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = PointMass(horizon=args.horizon, dt=args.dt)
    reference_coordinates = model.feasible_seed()
    np.savez_compressed(
        args.output_dir / "problem_definition.npz",
        start=model.start,
        goal=model.goal,
        workspace=model.workspace,
        obstacle_centers=np.asarray([obstacle.center for obstacle in model.obstacles]),
        obstacle_radii=np.asarray([obstacle.radius for obstacle in model.obstacles]),
        feasible_reference_x0=np.r_[model.start, np.zeros(2)],
        feasible_reference_coordinates=reference_coordinates,
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    results = [
        run_sampler(model, method, args, np.random.default_rng(child_seed))
        for method, child_seed in zip(METHODS, seed_sequence.spawn(len(METHODS)))
    ]
    rows = save_data(model, results, args.output_dir)
    plot_trajectories(model, results, args.output_dir)
    plot_performance(results, rows, args.output_dir)
    print_summary(results)
    print(f"\nSaved results to {args.output_dir.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
