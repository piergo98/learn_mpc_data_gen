"""Plot the recorded cart-pole timing breakdown as stacked bars.

The input is the ``timing_breakdown.csv`` written by
``cart_pole_horizon_sampling_comparison.py``: one row per MPC horizon and one
column per measured timing component.

For every horizon the figure contains one stacked bar per method:

* ``U``   Uniform: infeasible-candidate time below feasible-candidate time.
* ``B+P`` Bilevel + projection: bilevel NLPs, projection NLPs, boundary LPs,
  remaining pipeline time.
* ``S+P`` Slacked + projection: the same stack with slacked NLPs at the bottom.
* ``B+R`` Bilevel + rollout: bilevel NLPs below closed-loop rollout MPC time.
* ``S+R`` Slacked + rollout: slacked NLPs below closed-loop rollout MPC time.

A method is drawn only when all of its columns are present, so older CSVs
holding only the uniform and slacked-projection columns still plot.

The existing ``horizon_sweep.csv`` files contain only aggregate method times;
they cannot be used to reconstruct this breakdown.  The explicit schema and
validation here prevent an aggregate total from being presented as measured
component timing.

Example
-------
Run from the repository root::

    python3 src/plot_cart_pole_timing_breakdown.py \
        results/cart_pole_horizon_sampling_9/timing_breakdown.csv

By default, PNG and PDF files are written beside the input CSV.  Use
``--output`` to select a different output stem (without an extension) and
``--log`` for a logarithmic time axis.
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

# (short label, long label, stacked components from bottom to top)
METHODS = (
    ("U", "Uniform", ("uniform_infeasible_time_s", "uniform_feasible_time_s")),
    (
        "B+P",
        "Bilevel + projection",
        (
            "bilevel_problem_time_s",
            "bilevel_projected_problem_time_s",
            "bilevel_projection_lp_time_s",
            "bilevel_projection_other_time_s",
        ),
    ),
    (
        "S+P",
        "Slacked + projection",
        (
            "slacked_problem_time_s",
            "projected_problem_time_s",
            "projection_lp_time_s",
            "slacked_projection_other_time_s",
        ),
    ),
    (
        "B+R",
        "Bilevel + rollout",
        ("bilevel_rollout_active_set_exploration_time_s", "bilevel_rollout_time_s"),
    ),
    (
        "S+R",
        "Slacked + rollout",
        ("slacked_rollout_active_set_exploration_time_s", "slacked_rollout_time_s"),
    ),
)

# Components share a legend entry, and so a color, when they measure the same
# kind of work in different methods.
CATEGORY = {
    "uniform_infeasible_time_s": "uniform_infeasible",
    "uniform_feasible_time_s": "uniform_feasible",
    "bilevel_problem_time_s": "bilevel",
    "bilevel_rollout_active_set_exploration_time_s": "bilevel",
    "slacked_problem_time_s": "slacked",
    "slacked_rollout_active_set_exploration_time_s": "slacked",
    "bilevel_projected_problem_time_s": "projection_nlp",
    "projected_problem_time_s": "projection_nlp",
    "bilevel_projection_lp_time_s": "projection_lp",
    "projection_lp_time_s": "projection_lp",
    "bilevel_projection_other_time_s": "other",
    "slacked_projection_other_time_s": "other",
    "bilevel_rollout_time_s": "rollout",
    "slacked_rollout_time_s": "rollout",
}
LABELS = {
    "uniform_infeasible": "Infeasible uniform candidates",
    "uniform_feasible": "Feasible uniform candidates",
    "bilevel": "Bilevel problems",
    "slacked": "Slacked problems",
    "projection_nlp": "Projection problems",
    "projection_lp": "Projection LPs",
    "other": "Other pipeline",
    "rollout": "Closed-loop rollout MPC",
}
COLORS = {
    "uniform_infeasible": "#b7df8a",
    "uniform_feasible": "#33a02c",
    "bilevel": "#fb9a99",
    "slacked": "#6baed6",
    "projection_nlp": "#2171b5",
    "projection_lp": "#fdae6b",
    "other": "#bdbdbd",
    "rollout": "#6a3d9a",
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
        "--log",
        action="store_true",
        help="Use a logarithmic wall-time axis.",
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
        components = [
            column
            for _, _, method_columns in METHODS
            if set(method_columns) <= columns
            for column in method_columns
        ]
        missing = [] if HORIZON in columns and components else [
            column
            for column in (HORIZON, *METHODS[0][2])
            if column not in columns
        ]
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
            for column in components:
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
    rows: List[Dict[str, float]],
    output_stem: Path,
    title: str | None = None,
    log: bool = False,
) -> plt.Figure:
    """Create and save the grouped stacked timing-bar figure."""
    methods = [method for method in METHODS if all(c in rows[0] for c in method[2])]
    horizons = np.asarray([int(row[HORIZON]) for row in rows])
    positions = np.arange(len(rows), dtype=float)
    width = 0.8 / len(methods)

    fig_width = max(7.0, 0.5 * len(methods) * len(rows) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    labelled = set()
    for index, (short, _, columns) in enumerate(methods):
        offsets = positions + (index - (len(methods) - 1) / 2) * width
        bottom = np.zeros(len(rows))
        for column in columns:
            category = CATEGORY[column]
            values = _values(rows, column)
            ax.bar(
                offsets,
                values,
                width,
                bottom=bottom,
                label=None if category in labelled else LABELS[category],
                color=COLORS[category],
                edgecolor="white",
                linewidth=0.7,
            )
            labelled.add(category)
            bottom += values
        for offset, total in zip(offsets, bottom):
            ax.annotate(
                short,
                (offset, total),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(positions, [str(horizon) for horizon in horizons])
    ax.set_xlabel(r"MPC horizon $N$")
    ax.set_ylabel(r"Wall time $[\mathrm{s}]$")
    if log:
        ax.set_yscale("log")
    ax.margins(y=0.12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)

    # Spell out the bar annotations once instead of at every tick.
    ax.text(
        1.02,
        0.0,
        "\n".join(f"{short}: {long}" for short, long, _ in methods),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
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
    figure = plot_timings(rows, output_stem=output_stem, title=args.title, log=args.log)
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
