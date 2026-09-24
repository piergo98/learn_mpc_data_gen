"""Plot the recorded cart-pole timing breakdown as stacked bars.

The input is a CSV file with one row per MPC horizon and these columns::

    horizon,uniform_infeasible_time_s,uniform_feasible_time_s,\
    slacked_problem_time_s,projected_problem_time_s,projection_lp_time_s

For every horizon the figure contains two adjacent bars:

* ``Uniform`` stacks infeasible-candidate time below feasible-candidate time.
* ``Slacked + projection`` stacks the slacked NLP, nonlinear projection NLP,
  and Hit-and-Run boundary LP times in that order.

The existing ``horizon_sweep.csv`` files contain only aggregate method times;
they cannot be used to reconstruct this breakdown.  The explicit schema and
validation here prevent an aggregate total from being presented as measured
component timing.

Example
-------
Run from the repository root::

    python3 src/plot_cart_pole_timing_breakdown.py \
        results/cart_pole_horizon_sampling_7/timing_breakdown.csv

By default, PNG and PDF files are written beside the input CSV.  Use
``--output`` to select a different output stem (without an extension).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HORIZON = "horizon"
COMPONENTS = (
    "uniform_infeasible_time_s",
    "uniform_feasible_time_s",
    "slacked_problem_time_s",
    "projected_problem_time_s",
    "projection_lp_time_s",
)
REQUIRED_COLUMNS = (HORIZON, *COMPONENTS)

LABELS = {
    "uniform_infeasible_time_s": "Infeasible uniform candidates",
    "uniform_feasible_time_s": "Feasible uniform candidates",
    "slacked_problem_time_s": "Slacked problems",
    "projected_problem_time_s": "Projection problems",
    "projection_lp_time_s": "Projection LPs",
}
COLORS = {
    "uniform_infeasible_time_s": "#b7df8a",
    "uniform_feasible_time_s": "#33a02c",
    "slacked_problem_time_s": "#6baed6",
    "projected_problem_time_s": "#2171b5",
    "projection_lp_time_s": "#fdae6b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="CSV containing one measured timing-breakdown row per horizon.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path without an extension. Defaults to "
            "<input directory>/timing_breakdown_stacked."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def _finite_nonnegative(value: str, column: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: {column!r} must be a number, got {value!r}."
        ) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"Row {row_number}: {column!r} must be finite and non-negative, "
            f"got {value!r}."
        )
    return parsed


def load_timings(path: Path) -> List[Dict[str, float]]:
    """Load, validate, and horizon-sort measured timing components."""
    if not path.is_file():
        raise FileNotFoundError(f"Timing CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            aggregate_hint = ""
            if {"total_time_mean_s", "method"}.issubset(columns):
                aggregate_hint = (
                    " This appears to be an aggregate horizon_sweep.csv file; "
                    "its total_time_mean_s values do not identify the requested "
                    "timing components."
                )
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}."
                f"{aggregate_hint}"
            )

        rows: List[Dict[str, float]] = []
        seen_horizons = set()
        for row_number, raw in enumerate(reader, start=2):
            try:
                horizon_as_float = float(raw[HORIZON])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row {row_number}: 'horizon' must be an integer, "
                    f"got {raw[HORIZON]!r}."
                ) from exc
            horizon = int(horizon_as_float)
            if horizon_as_float != horizon or horizon <= 0:
                raise ValueError(
                    f"Row {row_number}: 'horizon' must be a positive integer, "
                    f"got {raw[HORIZON]!r}."
                )
            if horizon in seen_horizons:
                raise ValueError(f"Duplicate timing row for horizon {horizon}.")
            seen_horizons.add(horizon)

            row: Dict[str, float] = {HORIZON: horizon}
            for column in COMPONENTS:
                row[column] = _finite_nonnegative(
                    raw[column], column=column, row_number=row_number
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Timing CSV contains no data rows: {path}")
    return sorted(rows, key=lambda row: row[HORIZON])


def _values(rows: Iterable[Dict[str, float]], column: str) -> np.ndarray:
    return np.asarray([row[column] for row in rows], dtype=float)


def plot_timings(
    rows: List[Dict[str, float]], output_stem: Path, title: str | None = None
) -> plt.Figure:
    """Create and save the grouped stacked timing-bar figure."""
    horizons = np.asarray([int(row[HORIZON]) for row in rows])
    positions = np.arange(len(rows), dtype=float)
    width = 0.36

    fig_width = max(7.0, 0.95 * len(rows) + 2.7)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    uniform_bottom = np.zeros(len(rows))
    for column in COMPONENTS[:2]:
        values = _values(rows, column)
        ax.bar(
            positions - width / 2,
            values,
            width,
            bottom=uniform_bottom,
            label=LABELS[column],
            color=COLORS[column],
            edgecolor="white",
            linewidth=0.7,
        )
        uniform_bottom += values

    projection_bottom = np.zeros(len(rows))
    for column in COMPONENTS[2:]:
        values = _values(rows, column)
        ax.bar(
            positions + width / 2,
            values,
            width,
            bottom=projection_bottom,
            label=LABELS[column],
            color=COLORS[column],
            edgecolor="white",
            linewidth=0.7,
        )
        projection_bottom += values

    ax.set_xticks(positions, [str(horizon) for horizon in horizons])
    ax.set_xlabel(r"MPC horizon $N$")
    ax.set_ylabel(r"Wall time $[\mathrm{s}]$")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)

    # Name the two adjacent stacks without repeating long labels at every tick.
    ax.text(
        0.01,
        0.98,
        "left: Uniform    right: Slacked + projection",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def main() -> None:
    args = parse_args()
    rows = load_timings(args.input)
    output_stem = args.output or args.input.with_name("timing_breakdown_stacked")
    figure = plot_timings(rows, output_stem=output_stem, title=args.title)
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
