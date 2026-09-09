"""Compare MPC-feasible sampling strategies with supervised policy learning.

For each requested dataset size NB, this script:

1. Collects feasible rollout seeds with two strategies:
   - uniform rejection sampling over the state box,
   - the model's ``collect_feasible_samples`` method.
2. Runs closed-loop MPC for ``n_cl_steps`` from each seed and records
   ``(x_k, u_k^*)`` pairs, where ``u_k^*`` is the first MPC action.
3. Trains two equal-size ReLU MLP policies with MSE loss.
4. Evaluates both policies on the same validation dataset and stores summaries.

The procedure can be run for the linear MPC, the cart-pole NMPC, or both.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ULTRO_ROOT = Path(__file__).resolve().parent.parent
if str(ULTRO_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRO_ROOT))

from models.cart_pole import CartPole
from models.linear import LinearSystem
from src.linear_defaults import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_Q,
    DEFAULT_R,
    DEFAULT_U_BOUNDS,
    DEFAULT_X_BOUNDS,
)


DEFAULT_DATASET_SIZES = (100, 500, 1000, 5000, 10000)


@dataclass(frozen=True)
class SystemSpec:
    key: str
    label: str
    nx: int
    nu: int
    bounds: np.ndarray
    build_model: Callable[[], object]


@dataclass
class DatasetResult:
    system: str
    strategy: str
    NB: int
    n_seed_samples: int
    collection_time_s: float
    labeling_time_s: float
    train_time_s: float
    train_mse: float
    validation_mse: float
    validation_rmse: float
    validation_max_abs_error: float
    feasible_seed_count: int
    dataset_file: str
    model_file: str


class SupervisedMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@contextmanager
def maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    with redirect_stdout(io.StringIO()):
        yield


def build_linear_system(horizon: int = 10) -> LinearSystem:
    system = LinearSystem(
        DEFAULT_A.copy(),
        DEFAULT_B.copy(),
        DEFAULT_X_BOUNDS.copy(),
        DEFAULT_U_BOUNDS.copy(),
        dt=0.1,
        N=horizon,
    )
    system.define_simple_MPC_control(DEFAULT_Q.copy(), DEFAULT_R.copy())
    return system


def build_cart_pole(horizon: int = 20, dt_mpc: float = 0.1, dt_sim: float = 0.01) -> CartPole:
    return CartPole(horizon, dt_mpc=dt_mpc, dt_sim=dt_sim, sym_type="SX")


def make_system_specs(
    linear_horizon: int,
    cart_horizon: int,
    cart_dt_mpc: float,
    cart_dt_sim: float,
) -> dict[str, SystemSpec]:
    return {
        "linear": SystemSpec(
            key="linear",
            label="Linear MPC",
            nx=2,
            nu=1,
            bounds=DEFAULT_X_BOUNDS.copy(),
            build_model=lambda: build_linear_system(horizon=linear_horizon),
        ),
        "cart_pole": SystemSpec(
            key="cart_pole",
            label="Cart-pole NMPC",
            nx=4,
            nu=1,
            bounds=np.array([2.0, 5.0, np.pi, 10.0], dtype=float),
            build_model=lambda: build_cart_pole(
                horizon=cart_horizon,
                dt_mpc=cart_dt_mpc,
                dt_sim=cart_dt_sim,
            ),
        ),
    }


def normalize_samples(samples: np.ndarray, nx: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return np.zeros((0, nx), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {arr.shape}")
    if arr.shape[0] == nx and arr.shape[1] != nx:
        arr = arr.T
    if arr.shape[1] != nx:
        raise ValueError(f"samples must have {nx} columns, got shape {arr.shape}")
    return arr


def collect_uniform_feasible_samples(
    model,
    bounds: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> np.ndarray:
    samples = []
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    for _attempt in range(max_attempts):
        if len(samples) >= num_samples:
            break
        x0 = rng.uniform(low=-bounds, high=bounds)
        if model.is_feasible(x0):
            samples.append(x0)
    if len(samples) < num_samples:
        raise RuntimeError(
            f"Uniform sampler collected {len(samples)}/{num_samples} feasible samples "
            f"after {max_attempts} attempts."
        )
    return np.asarray(samples, dtype=float)


def collect_strategy_feasible_samples(
    system_key: str,
    model,
    bounds: np.ndarray,
    num_samples: int,
    max_attempts: int,
    quiet: bool,
) -> np.ndarray:
    with maybe_silence(quiet):
        if system_key == "cart_pole":
            samples = model.collect_feasible_samples(
                x_max=bounds,
                num_samples=num_samples,
                max_attempts=max_attempts,
                plot_results=False,
            )
        else:
            samples = model.collect_feasible_samples(
                num_samples=num_samples,
                max_attempts=max_attempts,
                plot_results=False,
            )
    samples = normalize_samples(samples, len(bounds))
    if samples.shape[0] < num_samples:
        raise RuntimeError(
            f"collect_feasible_samples returned {samples.shape[0]}/{num_samples} samples."
        )
    return samples[:num_samples]


def collect_rollout_seeds(
    system_key: str,
    model,
    strategy: str,
    bounds: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    max_attempts: int,
    quiet: bool,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if strategy == "uniform":
        samples = collect_uniform_feasible_samples(model, bounds, num_samples, rng, max_attempts)
    elif strategy == "collect_feasible_samples":
        samples = collect_strategy_feasible_samples(
            system_key, model, bounds, num_samples, max_attempts, quiet
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")
    return samples, time.perf_counter() - start


def solve_first_control(model, x0: np.ndarray) -> np.ndarray:
    u0 = model.solve_MPC(np.asarray(x0, dtype=float).reshape(-1), ret_seq=False)
    stats = model.solver.stats()
    if not bool(stats.get("success", False)):
        status = stats.get("return_status", stats.get("unified_return_status", "unknown"))
        raise RuntimeError(f"MPC solve failed with status '{status}' for x0={np.asarray(x0).reshape(-1)}")
    return np.asarray(u0, dtype=float).reshape(-1)


def checked_mpc_policy(model) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: solve_first_control(model, x)


def normalize_closed_loop_output(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    nx: int,
    nu: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_traj, dtype=float)
    u_arr = np.asarray(u_traj, dtype=float)

    if x_arr.ndim != 2:
        raise ValueError(f"Closed-loop states must be 2D, got shape {x_arr.shape}")
    if x_arr.shape[1] != nx and x_arr.shape[0] == nx:
        x_arr = x_arr.T
    if x_arr.shape[1] != nx:
        raise ValueError(f"Closed-loop states must have {nx} columns, got shape {x_arr.shape}")

    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(-1, nu)
    elif u_arr.ndim == 2 and u_arr.shape[1] != nu and u_arr.shape[0] == nu:
        u_arr = u_arr.T
    if u_arr.ndim != 2 or u_arr.shape[1] != nu:
        raise ValueError(f"Closed-loop controls must have {nu} columns, got shape {u_arr.shape}")

    return x_arr, u_arr


def build_labeled_dataset(
    model,
    seeds: np.ndarray,
    target_size: int,
    n_cl_steps: int,
    nx: int,
    nu: int,
    allow_partial: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_values = []
    u_values = []
    failed_rollouts = 0

    start = time.perf_counter()
    for seed in seeds:
        if len(x_values) >= target_size:
            break
        try:
            x_traj, u_traj = model.close_loop_simulation(
                np.asarray(seed, dtype=float).reshape(nx),
                Nsim=n_cl_steps,
                control_policy=checked_mpc_policy(model),
                plot_results=False,
            )
            x_rollout, u_rollout = normalize_closed_loop_output(x_traj, u_traj, nx, nu)
        except RuntimeError:
            failed_rollouts += 1
            continue

        n_pairs = min(n_cl_steps, u_rollout.shape[0], max(0, target_size - len(x_values)))
        x_values.extend(x_rollout[:n_pairs].copy())
        u_values.extend(u_rollout[:n_pairs].copy())

    if len(x_values) < target_size and not allow_partial:
        raise RuntimeError(
            f"Generated only {len(x_values)}/{target_size} labeled samples "
            f"({failed_rollouts} rollout failures)."
        )
    return (
        np.asarray(x_values[:target_size], dtype=float),
        np.asarray(u_values[:target_size], dtype=float),
        time.perf_counter() - start,
    )


def collect_labeled_dataset_with_retries(
    spec: SystemSpec,
    strategy: str,
    target_size: int,
    n_cl_steps: int,
    rng: np.random.Generator,
    max_attempts: int,
    quiet: bool,
    max_rounds: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Collect seeds and label them, retrying when closed-loop MPC fails.

    A seed can be MPC-feasible at time zero but lead to a later closed-loop
    state where the finite-horizon constrained MPC is infeasible. We keep the
    valid prefix from such rollouts and collect extra seeds until the requested
    number of labeled pairs is reached.
    """
    model = spec.build_model()
    X_chunks = []
    U_chunks = []
    seed_chunks = []
    collection_time = 0.0
    labeling_time = 0.0
    collected = 0

    for round_idx in range(max_rounds):
        remaining = target_size - collected
        if remaining <= 0:
            break

        n_seeds = max(1, int(np.ceil(remaining / n_cl_steps)))
        if round_idx > 0:
            n_seeds = max(n_seeds, int(np.ceil(0.25 * target_size / n_cl_steps)))

        seeds, round_collection_time = collect_rollout_seeds(
            spec.key,
            model,
            strategy,
            spec.bounds,
            n_seeds,
            rng,
            max_attempts=max_attempts,
            quiet=quiet,
        )
        X_part, U_part, round_labeling_time = build_labeled_dataset(
            model,
            seeds,
            target_size=remaining,
            n_cl_steps=n_cl_steps,
            nx=spec.nx,
            nu=spec.nu,
            allow_partial=True,
        )

        collection_time += round_collection_time
        labeling_time += round_labeling_time
        seed_chunks.append(seeds)
        if X_part.shape[0] > 0:
            X_chunks.append(X_part)
            U_chunks.append(U_part)
            collected += X_part.shape[0]

        if not quiet and X_part.shape[0] < remaining:
            print(
                f"  Labeled {collected}/{target_size} samples after retry round {round_idx + 1}."
            )

    if collected < target_size:
        raise RuntimeError(
            f"Generated only {collected}/{target_size} labeled samples after {max_rounds} retry rounds."
        )

    X = np.vstack(X_chunks)[:target_size]
    U = np.vstack(U_chunks)[:target_size]
    seeds_all = np.vstack(seed_chunks) if seed_chunks else np.zeros((0, spec.nx), dtype=float)
    return X, U, seeds_all, collection_time, labeling_time


def make_validation_dataset(
    spec: SystemSpec,
    size: int,
    n_cl_steps: int,
    rng: np.random.Generator,
    max_attempts: int,
    quiet: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X, U, _seeds, _collection_time, _labeling_time = collect_labeled_dataset_with_retries(
        spec,
        "uniform",
        size,
        n_cl_steps,
        rng,
        max_attempts,
        quiet,
    )
    return X, U


def standardize_train_validation(
    X_train: np.ndarray,
    U_train: np.ndarray,
    X_validation: np.ndarray,
    U_validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0

    u_mean = U_train.mean(axis=0)
    u_std = U_train.std(axis=0)
    u_std[u_std < 1e-8] = 1.0

    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "u_mean": u_mean.tolist(),
        "u_std": u_std.tolist(),
    }
    return (
        (X_train - x_mean) / x_std,
        (U_train - u_mean) / u_std,
        (X_validation - x_mean) / x_std,
        (U_validation - u_mean) / u_std,
        stats,
    )


def train_supervised_mlp(
    X_train: np.ndarray,
    U_train: np.ndarray,
    X_validation: np.ndarray,
    U_validation: np.ndarray,
    hidden_sizes: tuple[int, ...],
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    early_stopping_patience: int = 25,
    early_stopping_min_delta: float = 1e-6,
) -> tuple[SupervisedMLP, dict[str, float | list[float]]]:
    torch.manual_seed(seed)
    Xs_train, Us_train, Xs_validation, Us_validation, stats = standardize_train_validation(
        X_train, U_train, X_validation, U_validation
    )

    model = SupervisedMLP(X_train.shape[1], U_train.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(
        torch.as_tensor(Xs_train, dtype=torch.float32),
        torch.as_tensor(Us_train, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        generator=generator,
    )

    start = time.perf_counter()
    model.train()
    best_state_dict = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        best_validation_loss = float(
            loss_fn(
                model(torch.as_tensor(Xs_validation, dtype=torch.float32, device=device)),
                torch.as_tensor(Us_validation, dtype=torch.float32, device=device),
            ).item()
        )
    model.train()
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = epochs

    for _epoch in range(epochs):
        for xb, ub in loader:
            xb = xb.to(device)
            ub = ub.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), ub)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(
                    model(torch.as_tensor(Xs_validation, dtype=torch.float32, device=device)),
                    torch.as_tensor(Us_validation, dtype=torch.float32, device=device),
                ).item()
            )
        model.train()

        print(
            f"Epoch {_epoch + 1}/{epochs}, Batch Loss: {loss.item():.6f}, "
            f"Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = _epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                stopped_epoch = _epoch + 1
                print(
                    f"Early stopping at epoch {stopped_epoch}: "
                    f"no validation improvement for {early_stopping_patience} epochs."
                )
                break

    train_time = time.perf_counter() - start

    model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(torch.as_tensor(Xs_train, dtype=torch.float32, device=device))
        validation_pred_scaled = model(torch.as_tensor(Xs_validation, dtype=torch.float32, device=device))

    u_mean = np.asarray(stats["u_mean"], dtype=float)
    u_std = np.asarray(stats["u_std"], dtype=float)
    train_pred = train_pred_scaled.cpu().numpy() * u_std + u_mean
    validation_pred = validation_pred_scaled.cpu().numpy() * u_std + u_mean

    train_err = train_pred - U_train
    validation_err = validation_pred - U_validation
    metrics: dict[str, float | list[float]] = {
        "train_time_s": train_time,
        "train_mse": float(np.mean(train_err**2)),
        "validation_mse": float(np.mean(validation_err**2)),
        "validation_rmse": float(np.sqrt(np.mean(validation_err**2))),
        "validation_max_abs_error": float(np.max(np.abs(validation_err))),
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": float(best_epoch),
        "stopped_epoch": float(stopped_epoch),
        "early_stopping_patience": float(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        **stats,
    }
    return model, metrics


def save_results(results: list[DatasetResult], output_dir: Path) -> None:
    json_path = output_dir / "sampling_strategy_supervised_results.json"
    csv_path = output_dir / "sampling_strategy_supervised_summary.csv"
    rows = [asdict(row) for row in results]

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    if not rows:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: list[DatasetResult], output_dir: Path) -> None:
    for system in sorted({row.system for row in results}):
        rows = [row for row in results if row.system == system]
        sizes = sorted({row.NB for row in rows})
        strategies = ["uniform", "collect_feasible_samples"]
        labels = {"uniform": "Uniform feasible", "collect_feasible_samples": "collect_feasible_samples"}

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        width = 0.36
        x = np.arange(len(sizes))

        for idx, strategy in enumerate(strategies):
            strategy_rows = {
                row.NB: row for row in rows if row.strategy == strategy
            }
            mse = [strategy_rows[size].validation_mse if size in strategy_rows else np.nan for size in sizes]
            dataset_build_time = [
                (
                    strategy_rows[size].collection_time_s
                    + strategy_rows[size].labeling_time_s
                )
                if size in strategy_rows
                else np.nan
                for size in sizes
            ]
            axes[0].plot(sizes, mse, marker="o", linewidth=2, label=labels[strategy])
            axes[1].bar(
                x + (idx - 0.5) * width,
                dataset_build_time,
                width,
                label=labels[strategy],
            )

        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Dataset size NB")
        axes[0].set_ylabel("Validation MSE")
        axes[0].set_title(f"{system}: supervised policy error")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_yscale("log")
        axes[1].set_xlabel("Dataset size NB")
        axes[1].set_ylabel("Dataset build time [s]")
        axes[1].set_title(f"{system}: dataset construction")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(size) for size in sizes])
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"{system}_sampling_strategy_supervised_comparison.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for idx, strategy in enumerate(strategies):
            strategy_rows = {
                row.NB: row for row in rows if row.strategy == strategy
            }
            train_times = [
                strategy_rows[size].train_time_s if size in strategy_rows else np.nan
                for size in sizes
            ]
            train_mse = [
                strategy_rows[size].train_mse if size in strategy_rows else np.nan
                for size in sizes
            ]
            validation_mse = [
                strategy_rows[size].validation_mse if size in strategy_rows else np.nan
                for size in sizes
            ]

            axes[0].bar(
                x + (idx - 0.5) * width,
                train_times,
                width,
                label=labels[strategy],
            )
            axes[1].plot(
                sizes,
                train_mse,
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"{labels[strategy]} train",
            )
            axes[1].plot(
                sizes,
                validation_mse,
                marker="s",
                linewidth=2,
                linestyle="--",
                label=f"{labels[strategy]} validation",
            )

        axes[0].set_yscale("log")
        axes[0].set_xlabel("Dataset size NB")
        axes[0].set_ylabel("Training time [s]")
        axes[0].set_title(f"{system}: neural-network training time")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(size) for size in sizes])
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend()

        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Dataset size NB")
        axes[1].set_ylabel("MSE")
        axes[1].set_title(f"{system}: training and validation errors")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"{system}_training_time_and_errors.png", dpi=200)
        plt.close(fig)


def histogram_l1_distance(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")

    lower = float(min(np.min(a), np.min(b)))
    upper = float(max(np.max(a), np.max(b)))
    if np.isclose(lower, upper):
        return 0.0

    hist_a, edges = np.histogram(a, bins=bins, range=(lower, upper), density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)
    prob_a = hist_a / max(1, np.sum(hist_a))
    prob_b = hist_b / max(1, np.sum(hist_b))
    return float(0.5 * np.sum(np.abs(prob_a - prob_b)))


def describe_array_columns(values: np.ndarray) -> list[dict[str, float]]:
    values = np.asarray(values, dtype=float)
    return [
        {
            "mean": float(np.mean(values[:, idx])),
            "std": float(np.std(values[:, idx])),
            "min": float(np.min(values[:, idx])),
            "max": float(np.max(values[:, idx])),
        }
        for idx in range(values.shape[1])
    ]


def save_sample_difference_summary(
    spec: SystemSpec,
    NB: int,
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    X_validation: np.ndarray,
    U_validation: np.ndarray,
    output_dir: Path,
) -> None:
    uniform_X, uniform_U, uniform_seeds = datasets["uniform"]
    informative_X, informative_U, informative_seeds = datasets["collect_feasible_samples"]

    summary = {
        "system": spec.key,
        "NB": int(NB),
        "state_labels": [f"x[{idx}]" for idx in range(spec.nx)],
        "control_labels": [f"u[{idx}]" for idx in range(spec.nu)],
        "uniform": {
            "X": describe_array_columns(uniform_X),
            "U": describe_array_columns(uniform_U),
            "seeds": describe_array_columns(uniform_seeds),
        },
        "collect_feasible_samples": {
            "X": describe_array_columns(informative_X),
            "U": describe_array_columns(informative_U),
            "seeds": describe_array_columns(informative_seeds),
        },
        "validation": {
            "X": describe_array_columns(X_validation),
            "U": describe_array_columns(U_validation),
        },
        "histogram_l1_distance": {
            "uniform_vs_collect_feasible_samples_X": [
                histogram_l1_distance(uniform_X[:, idx], informative_X[:, idx])
                for idx in range(spec.nx)
            ],
            "uniform_vs_collect_feasible_samples_U": [
                histogram_l1_distance(uniform_U[:, idx], informative_U[:, idx])
                for idx in range(spec.nu)
            ],
            "validation_vs_uniform_X": [
                histogram_l1_distance(X_validation[:, idx], uniform_X[:, idx])
                for idx in range(spec.nx)
            ],
            "validation_vs_collect_feasible_samples_X": [
                histogram_l1_distance(X_validation[:, idx], informative_X[:, idx])
                for idx in range(spec.nx)
            ],
        },
    }

    output_path = output_dir / f"{spec.key}_NB{NB}_sample_difference_summary.json"
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)


def plot_sample_difference_diagnostics(
    spec: SystemSpec,
    NB: int,
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    X_validation: np.ndarray,
    U_validation: np.ndarray,
    output_dir: Path,
) -> Path:
    labels = {
        "uniform": "Uniform train",
        "collect_feasible_samples": "Informative train",
        "validation": "Validation",
    }
    colors = {
        "uniform": "tab:blue",
        "collect_feasible_samples": "tab:orange",
        "validation": "tab:green",
    }

    n_rows = spec.nx + spec.nu
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.35 * n_rows))
    axes = np.atleast_2d(axes)

    values_by_label = {
        "uniform": datasets["uniform"][:2],
        "collect_feasible_samples": datasets["collect_feasible_samples"][:2],
        "validation": (X_validation, U_validation),
    }

    for state_idx in range(spec.nx):
        ax_hist = axes[state_idx, 0]
        ax_cdf = axes[state_idx, 1]
        for key, (X, _U) in values_by_label.items():
            values = X[:, state_idx]
            ax_hist.hist(values, bins=40, density=True, alpha=0.35, color=colors[key], label=labels[key])
            sorted_values = np.sort(values)
            cdf = np.linspace(0.0, 1.0, sorted_values.size, endpoint=False)
            ax_cdf.plot(sorted_values, cdf, color=colors[key], linewidth=2, label=labels[key])
        ax_hist.set_ylabel(f"x[{state_idx}]")
        ax_hist.grid(True, alpha=0.3)
        ax_cdf.grid(True, alpha=0.3)

    for control_idx in range(spec.nu):
        row = spec.nx + control_idx
        ax_hist = axes[row, 0]
        ax_cdf = axes[row, 1]
        for key, (_X, U) in values_by_label.items():
            values = U[:, control_idx]
            ax_hist.hist(values, bins=40, density=True, alpha=0.35, color=colors[key], label=labels[key])
            sorted_values = np.sort(values)
            cdf = np.linspace(0.0, 1.0, sorted_values.size, endpoint=False)
            ax_cdf.plot(sorted_values, cdf, color=colors[key], linewidth=2, label=labels[key])
        ax_hist.set_ylabel(f"u[{control_idx}]")
        ax_hist.grid(True, alpha=0.3)
        ax_cdf.grid(True, alpha=0.3)

    axes[0, 0].set_title(f"{spec.label}: marginal densities, NB={NB}")
    axes[0, 1].set_title("Empirical CDFs")
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Value")
    axes[-1, 1].set_xlabel("Value")

    fig.tight_layout()
    output_path = output_dir / f"{spec.key}_NB{NB}_sample_difference_diagnostics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    if spec.nx >= 2:
        rng = np.random.default_rng(1234 + NB)
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for key, (X, _U) in values_by_label.items():
            n_plot = min(1200, X.shape[0])
            idx = rng.choice(X.shape[0], size=n_plot, replace=False)
            ax.scatter(
                X[idx, 0],
                X[idx, 1],
                s=12,
                alpha=0.45,
                color=colors[key],
                label=labels[key],
            )
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title(f"{spec.label}: state samples projected onto x[0], x[1], NB={NB}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        scatter_path = output_dir / f"{spec.key}_NB{NB}_sample_projection.png"
        fig.savefig(scatter_path, dpi=200)
        plt.close(fig)

    save_sample_difference_summary(spec, NB, datasets, X_validation, U_validation, output_dir)
    return output_path


def evaluate_torch_policy(
    policy: SupervisedMLP,
    metrics: dict[str, float | list[float]],
    x: np.ndarray,
    device: str,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(1, -1)
    x_mean = np.asarray(metrics["x_mean"], dtype=float).reshape(1, -1)
    x_std = np.asarray(metrics["x_std"], dtype=float).reshape(1, -1)
    u_mean = np.asarray(metrics["u_mean"], dtype=float).reshape(1, -1)
    u_std = np.asarray(metrics["u_std"], dtype=float).reshape(1, -1)

    x_scaled = (x_arr - x_mean) / x_std
    policy.eval()
    with torch.no_grad():
        u_scaled = policy(torch.as_tensor(x_scaled, dtype=torch.float32, device=device))
    return (u_scaled.cpu().numpy() * u_std + u_mean).reshape(-1)


def run_model_closed_loop(
    model,
    x0: np.ndarray,
    n_steps: int,
    nx: int,
    nu: int,
    control_policy: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_traj, u_traj = model.close_loop_simulation(
        np.asarray(x0, dtype=float).reshape(nx),
        Nsim=n_steps,
        control_policy=control_policy,
        plot_results=False,
    )
    return normalize_closed_loop_output(x_traj, u_traj, nx, nu)


def plot_closed_loop_policy_test(
    spec: SystemSpec,
    policies: dict[str, tuple[SupervisedMLP, dict[str, float | list[float]]]],
    x0: np.ndarray,
    n_steps: int,
    NB: int,
    device: str,
    output_dir: Path,
) -> Path:
    model = spec.build_model()
    trajectories: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    try:
        trajectories["MPC"] = run_model_closed_loop(
            model,
            x0,
            n_steps,
            spec.nx,
            spec.nu,
            control_policy=checked_mpc_policy(model),
        )
    except RuntimeError as exc:
        print(f"Skipping MPC closed-loop reference plot for NB={NB}: {exc}")

    labels = {
        "uniform": "NN trained on uniform feasible",
        "collect_feasible_samples": "NN trained on collect_feasible_samples",
    }
    for strategy, (policy, metrics) in policies.items():
        trajectories[labels[strategy]] = run_model_closed_loop(
            model,
            x0,
            n_steps,
            spec.nx,
            spec.nu,
            control_policy=lambda x, p=policy, m=metrics: evaluate_torch_policy(p, m, x, device),
        )

    dt = float(getattr(model, "dt_mpc", getattr(model, "dt", 1.0)))
    time_x = np.arange(n_steps + 1) * dt
    time_u = np.arange(n_steps) * dt
    state_labels = (
        ["Position", "Velocity", "Angle", "Angular velocity"]
        if spec.nx == 4
        else [f"x[{idx}]" for idx in range(spec.nx)]
    )

    fig, axes = plt.subplots(spec.nx + 1, 1, figsize=(11, 2.2 * (spec.nx + 1)), sharex=True)
    axes = np.atleast_1d(axes)

    for label, (x_traj, _u_traj) in trajectories.items():
        for state_idx in range(spec.nx):
            axes[state_idx].plot(time_x, x_traj[:, state_idx], linewidth=2, label=label)

    for state_idx in range(spec.nx):
        axes[state_idx].set_ylabel(state_labels[state_idx])
        axes[state_idx].grid(True, alpha=0.3)
        if state_idx == 0:
            axes[state_idx].set_title(
                f"{spec.label}: closed-loop test from same x0, NB={NB}"
            )

    for label, (_x_traj, u_traj) in trajectories.items():
        axes[-1].step(time_u, u_traj[:, 0], where="post", linewidth=2, label=label)
    axes[-1].set_ylabel("u[0]")
    axes[-1].set_xlabel("Time [s]")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend(loc="best")

    fig.tight_layout()
    output_path = output_dir / f"{spec.key}_NB{NB}_closed_loop_policy_test.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def run_experiment(args: argparse.Namespace) -> tuple[list[DatasetResult], Path]:
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = ULTRO_ROOT / "results" / f"sampling_strategy_supervised_{stamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    specs = make_system_specs(
        linear_horizon=args.linear_horizon,
        cart_horizon=args.cart_horizon,
        cart_dt_mpc=args.cart_dt_mpc,
        cart_dt_sim=args.cart_dt_sim,
    )

    selected_systems = list(specs) if args.system == "both" else [args.system]
    results: list[DatasetResult] = []

    for system_key in selected_systems:
        spec = specs[system_key]
        print(f"\n=== {spec.label} ===")
        validation_rng = np.random.default_rng(args.seed + 10_000)
        print(f"Building shared validation dataset with {args.validation_size} samples...")
        X_validation, U_validation = make_validation_dataset(
            spec,
            size=args.validation_size,
            n_cl_steps=args.n_cl_steps,
            rng=validation_rng,
            max_attempts=args.uniform_max_attempts,
            quiet=args.quiet,
        )
        np.savez_compressed(
            output_dir / f"{system_key}_validation.npz",
            X=X_validation,
            U=U_validation,
        )

        for NB in args.dataset_sizes:
            n_seeds = int(np.ceil(NB / args.n_cl_steps))
            trained_policies: dict[str, tuple[SupervisedMLP, dict[str, float | list[float]]]] = {}
            generated_datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for strategy_idx, strategy in enumerate(("uniform", "collect_feasible_samples")):
                print(f"\n{spec.label}: strategy={strategy}, NB={NB}, initial seed target={n_seeds}")
                rng = np.random.default_rng(args.seed + NB * 17 + strategy_idx)
                max_attempts = (
                    args.uniform_max_attempts
                    if strategy == "uniform"
                    else args.strategy_max_attempts
                )
                X_train, U_train, seeds, collection_time, labeling_time = collect_labeled_dataset_with_retries(
                    spec,
                    strategy,
                    NB,
                    args.n_cl_steps,
                    rng,
                    max_attempts,
                    args.quiet,
                )

                dataset_file = f"{system_key}_{strategy}_NB{NB}.npz"
                np.savez_compressed(
                    output_dir / dataset_file,
                    X=X_train,
                    U=U_train,
                    seeds=seeds,
                )
                generated_datasets[strategy] = (X_train, U_train, seeds)

                policy, metrics = train_supervised_mlp(
                    X_train,
                    U_train,
                    X_validation,
                    U_validation,
                    hidden_sizes=tuple(args.hidden_sizes),
                    seed=args.seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_min_delta=args.early_stopping_min_delta,
                )

                model_file = f"{system_key}_{strategy}_NB{NB}_policy.pt"
                torch.save(
                    {
                        "state_dict": policy.state_dict(),
                        "system": system_key,
                        "strategy": strategy,
                        "NB": NB,
                        "input_size": spec.nx,
                        "output_size": spec.nu,
                        "hidden_sizes": tuple(args.hidden_sizes),
                        "metrics": metrics,
                    },
                    output_dir / model_file,
                )
                trained_policies[strategy] = (policy, metrics)

                result = DatasetResult(
                    system=system_key,
                    strategy=strategy,
                    NB=NB,
                    n_seed_samples=int(seeds.shape[0]),
                    collection_time_s=float(collection_time),
                    labeling_time_s=float(labeling_time),
                    train_time_s=float(metrics["train_time_s"]),
                    train_mse=float(metrics["train_mse"]),
                    validation_mse=float(metrics["validation_mse"]),
                    validation_rmse=float(metrics["validation_rmse"]),
                    validation_max_abs_error=float(metrics["validation_max_abs_error"]),
                    feasible_seed_count=int(seeds.shape[0]),
                    dataset_file=dataset_file,
                    model_file=model_file,
                )
                results.append(result)
                save_results(results, output_dir)
                plot_results(results, output_dir)
                print(
                    f"Done: collection={collection_time:.2f}s, labeling={labeling_time:.2f}s, "
                    f"train MSE={result.train_mse:.3e}, validation MSE={result.validation_mse:.3e}"
                )

            if len(generated_datasets) == 2:
                diagnostics_path = plot_sample_difference_diagnostics(
                    spec=spec,
                    NB=NB,
                    datasets=generated_datasets,
                    X_validation=X_validation,
                    U_validation=U_validation,
                    output_dir=output_dir,
                )
                print(f"Saved sample difference diagnostics: {diagnostics_path.name}")

            if args.closed_loop_test_steps > 0 and len(trained_policies) == 2:
                plot_path = plot_closed_loop_policy_test(
                    spec=spec,
                    policies=trained_policies,
                    x0=X_validation[0],
                    n_steps=args.closed_loop_test_steps,
                    NB=NB,
                    device=args.device,
                    output_dir=output_dir,
                )
                print(f"Saved closed-loop policy test plot: {plot_path.name}")

    return results, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare uniform feasible sampling with collect_feasible_samples for supervised MPC imitation."
    )
    parser.add_argument("--system", choices=["linear", "cart_pole", "both"], default="both")
    parser.add_argument("--dataset-sizes", type=int, nargs="+", default=list(DEFAULT_DATASET_SIZES))
    parser.add_argument("--n-cl-steps", type=int, default=10)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[6, 6])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-horizon", type=int, default=10)
    parser.add_argument("--cart-horizon", type=int, default=10)
    parser.add_argument("--cart-dt-mpc", type=float, default=0.1)
    parser.add_argument("--cart-dt-sim", type=float, default=0.01)
    parser.add_argument("--uniform-max-attempts", type=int, default=200000)
    parser.add_argument("--strategy-max-attempts", type=int, default=20000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--closed-loop-test-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose sampler output.")
    return parser.parse_args()


if __name__ == "__main__":
    experiment_results, experiment_dir = run_experiment(parse_args())
    print(f"\nSaved {len(experiment_results)} experiment rows in {experiment_dir}")
