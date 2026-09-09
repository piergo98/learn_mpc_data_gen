# Sampling — feasible initial-state sampling for MPC

Code for the **sampling** contribution: how to draw initial states `x0` that
are *feasible* for a (possibly nonconvex, possibly discontinuous) MPC
problem, and how the different strategies compare in cost, coverage, and
rejection behaviour.

The package is self-contained: `models/` and `src/` carry everything the
experiments need, with no dependency on an external `ultro` package or other
repository.

---

## Why sampling is a problem at all

Training or benchmarking an MPC policy needs initial states for which the MPC
is actually solvable. Uniform sampling over a state box wastes almost all of
its candidates once the feasible set is a thin, nonconvex sliver of that box.
The strategies collected here replace rejection with *construction*: they solve
an optimization problem whose solution is feasible by design, or they walk a
chain that stays inside a local convex approximation of the feasible set and
then project back onto the nonlinear one.

## The strategies

| Name | Model method | Idea |
| --- | --- | --- |
| **uniform** | `is_feasible` + rejection | Draw from the state box, keep it if the MPC solves. The baseline. |
| **slacked** | `seek_x0`, `collect_feasible_samples_with_slacked` | Add slacks to the MPC residuals with randomly *spiked* weights (`spiky_residual_weights`) and minimize them. Different random weights push the solution to different corners of the feasible set. Quadratic or L1 penalty. |
| **bilevel** | `bilevel_active_set`, `collect_feasible_samples_with_bilevel` | Solve a bilevel problem whose lower level is the MPC KKT system with explicit complementarity constraints. Every solution is feasible by construction, so this sampler cannot return an infeasible state. Needs the CasADi `ccopt` plugin. |
| **Hit-and-Run** | `collect_linearized_hit_and_run_samples`, `sample_hit_and_run` | Linearize the OCP at a feasible seed, run a Hit-and-Run chain inside the resulting polytope (LPs solved with `qpoases`), then map each proposal back with `project_onto_feasible_set`. |
| **hybrid / seeded HR** | combination | Hit-and-Run seeded from bilevel solutions instead of uniform ones — the seed quality is the only difference. |
| **rollout** | `solve_MPC` / `run_closed_loop_batch`, driven by the scripts | Propagate a few good seeds with closed-loop MPC and keep every visited state. Cheap per sample, but samples are correlated along trajectories. Implemented in the experiment scripts (`collect_bilevel_rollout`), not as a model method — only `CartPole` exposes a batched `run_closed_loop_batch`. |

Not every model implements every strategy — see the table in
[Models](#models).

---

## Layout

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── models/                 # dynamical systems + their samplers
│   ├── cart_pole.py            -> CartPole
│   ├── discontinuous_control.py-> DiscontinuousControl
│   ├── linear.py               -> LinearSystem
│   └── point_mass.py           -> PointMass, CircularObstacle
├── src/                    # executable comparison experiments
│   ├── cart_pole_feasible_sampling_comparison.py
│   ├── cart_pole_hit_and_run_seed_comparison.py
│   ├── cart_pole_horizon_sampling_comparison.py
│   ├── discontinuous_control_sampling_comparison.py
│   ├── discontinuous_control_projection_sampling_comparison.py
│   ├── point_mass_obstacle_sampling_comparison.py
│   ├── sampling_strategy_supervised_comparison.py
│   └── linear_defaults.py      # default linear-MPC problem data
└── results/                # experiment output (gitignored)
```

Each script prepends the repository root to `sys.path` at import time, so
`from models.cart_pole import CartPole` resolves to `models/` at the top of
this repository regardless of the current working directory. Default output
directories resolve to `results/`.

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install .
```

This installs the pinned dependencies from `requirements.txt` together with
the `models` and `src` packages. For an editable checkout during development,
use `pip install -e .` instead.

`torch` is only imported by `sampling_strategy_supervised_comparison.py`; every
other script runs without it.

### CasADi and solver plugins

The models build their problems with CasADi and need three solver plugins:

| Plugin | Used by | In the PyPI wheel? |
| --- | --- | --- |
| `ipopt` (`nlpsol`) | nonlinear MPC, seed problems, slacked and projection problems | yes |
| `qpoases` (`conic`) | linear programs of the Hit-and-Run chain | yes |
| `ccopt` (`nlpsol`) | complementarity relaxation of the bilevel active-set sampler | **no** |

The **bilevel** sampler therefore requires a CasADi build with the `ccopt`
plugin, and that build must be recent enough to accept the `cc_pairs` option.
Point the interpreter at it before running:

```bash
export PYTHONPATH=/path/to/casadi-ccopt/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/casadi-ccopt/lib:$LD_LIBRARY_PATH
```

Everything that does not use the bilevel sampler runs on the stock wheel.

---

## Running the experiments

All commands are run **from the repository root**. Every script takes
`--help`, `--seed`, and `--output-dir`, and writes a CSV/JSON summary plus PNG
and PDF figures. `--show` opens the figures instead of only saving them.

### 1. Point mass in an obstacle course

A planar double integrator must reach a goal with zero velocity through five
walls of circular obstacles — a strongly nonconvex feasible set. Compares
uniform rejection, quadratic residual slacks, linear (L1) residual slacks, and
linearized Hit-and-Run.

```bash
python3 src/point_mass_obstacle_sampling_comparison.py --num-samples 12
```

Key options: `--horizon`, `--dt`, `--max-attempts`, `--hr-num-seeds`,
`--hr-burn-in`, `--hr-thinning`, `--hr-lp-solver`, `--no-hr-projection`.
Output: `problem_definition.npz`, `samples_<method>.npz`, `summary.csv`,
`trajectory_comparison.png`, `sampling_performance.png` in
`results/point_mass_obstacle_sampling/`.

### 2. Discontinuous control — sampler comparison

Uniform rejection vs. the bilevel active-set sampler vs. linearized
Hit-and-Run with nonlinear projection, on a 2-state model with discontinuous
dynamics whose exact feasible set is known for `N = 2`.

```bash
python3 src/discontinuous_control_sampling_comparison.py --num-samples 100
```

Key options: `--horizon`, `--state-bound`, `--hr-num-seeds`,
`--hr-projection-weight`, `--verbose`.
Output: `sampling_statistics.{csv,json}`, `sampled_initial_states.png`,
`sampling_performance.png` in `results/discontinuous_control_sampling/`.

### 3. Discontinuous control — projection vs. rollout post-processing

The paper-facing version of the previous experiment. Compares `uniform`,
`bilevel_projection`, `slacked_projection`, and `bilevel_rollout`, and splits
the wall-clock time into a generation and a post-processing phase. Rejections
are reported *by cause* rather than as a single rate, because the bilevel
sampler cannot produce an infeasible state and can only lose a candidate for
other reasons.

```bash
python3 src/discontinuous_control_projection_sampling_comparison.py \
    --num-samples 50 --repeats 3
```

Key options: `--horizon`, `--state-bound`, `--projection-weight`,
`--slack-penalty {quadratic,l1}`, `--close-to-bounds/--no-close-to-bounds`,
`--rollout-steps`, `--rollout-seeds`, `--rollout-dedupe`, `--rollout-tol`,
`--linearization-points`, `--mpc-restarts`, `--repeats`.
Output: `paper_summary.csv`, `sampling_statistics.{csv,json}`,
`samples_<method>.csv`, `candidates_<method>.csv`, `rollout_trajectories.csv`,
`linearized_sets.csv`, `feasible_set_boundary.csv`, `post_processing_effect.png`,
`linearized_feasible_set.png`, `sampled_initial_states.png`,
`sampling_performance.png` in `results/discontinuous_control_projection_sampling/`.

### 4. Cart-pole — feasible sampling comparison

Four-state nonlinear model. Compares the bilevel sampler, uniform rejection,
local linearized Hit-and-Run with projection, and bilevel-seeded Hit-and-Run
(on by default). The quadratic- and L1-slack variants can be added with
`--slack-samplers`.

```bash
python3 src/cart_pole_feasible_sampling_comparison.py --num-samples 100
```

Key options: `--horizon`, `--dt`, `--bounds-scale`, `--max-attempts`,
`--hybrid-bilevel-hit-and-run/--no-...`, `--slack-samplers/--no-slack-samplers`,
`--hr-num-seeds`, `--hr-projection-weight`, `--verbose-collector`.
Output: `summary.csv`, `samples_<method>.csv`, `uniform_candidates.csv`,
`linearized_hr_candidates.csv`, `hybrid_bilevel_hr_candidates.csv`,
`pairwise_feasible_samples.png`, `state_marginals.png`,
`uniform_feasibility_checks.png`, `sampling_performance.png` in
`results/cart_pole_feasible_sampling/`.

### 5. Cart-pole — does the Hit-and-Run seed matter?

Same Hit-and-Run and projection settings in both runs; the only difference is
whether the seeds come from uniform rejection or from the bilevel sampler.

```bash
python3 src/cart_pole_hit_and_run_seed_comparison.py --num-samples 100
```

Key options: `--horizon`, `--dt`, `--num-seeds`, `--projection-weight`,
`--bounds-scale`, `--max-attempts`, `--verbose`.
Output: `summary.csv`, `projected_samples_pairwise.png`,
`projected_samples_3d.png`, `performance_comparison.png` in
`results/cart_pole_hit_and_run_seed_comparison/`.

### 6. Cart-pole — effect of the MPC horizon

Runs the four samplers of experiment 3 once per horizon, so the cost of data
generation can be read as a function of `N`.

```bash
python3 src/cart_pole_horizon_sampling_comparison.py \
    --horizons 5 10 15 20 --num-samples 50 --repeats 3
```

Key options: same post-processing flags as experiment 3, plus `--horizons` and
`--dt`.
Output: `horizon_sweep.{csv,json}`, `samples_N<h>_<method>.csv`,
`horizon_trends.png` in `results/cart_pole_horizon_sampling/`.

### 7. Does the sampler change what a policy learns?

Ties sampling to a downstream task. For each dataset size `NB`: collect seeds
with uniform rejection and with the model's `collect_feasible_samples`, run
closed-loop MPC from each seed to label `(x_k, u_k*)` pairs, train two
identically sized ReLU MLPs, and compare their validation error. Runs on the
linear MPC, the cart-pole NMPC, or both. **Requires `torch`.**

```bash
python3 src/sampling_strategy_supervised_comparison.py --help
```

Output: `config.json`, `sampling_strategy_supervised_summary.csv`,
`sampling_strategy_supervised_results.json` in a timestamped
`results/sampling_strategy_supervised_<YYYY-MM-DD_HH-MM-SS>/`.

---

## Models

| Model | Class | uniform | slacked | bilevel | Hit-and-Run | rollout |
| --- | --- | :-: | :-: | :-: | :-: | :-: |
| `models/point_mass.py` | `PointMass` | ✓ | ✓ | — | ✓ | — |
| `models/discontinuous_control.py` | `DiscontinuousControl` | ✓ | ✓ | ✓ | ✓ | script |
| `models/cart_pole.py` | `CartPole` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `models/linear.py` | `LinearSystem` | ✓ | ✓ | — | — | script |

"script" means the strategy is driven from the experiment file rather than
provided as a model method.

Shared sampler entry points:

```python
model.is_feasible(x0)                             # rejection test
model.seek_x0(...)                                # slacked extreme-state problem
model.spiky_residual_weights(...)                 # random spiked slack weights
model.bilevel_active_set(x_max, ...)              # bilevel active-set problem
model.collect_feasible_samples_with_slacked(...)  # slacked sampler, batched
model.collect_feasible_samples_with_bilevel(...)  # bilevel sampler, batched
model.collect_linearized_hit_and_run_samples(...) # HR chain + projection
model.project_onto_feasible_set(x_candidate, ...) # nonlinear projection
model.plot_feasible_set(...)                      # 2-D feasible-set diagnostics
```

---

## Using a model directly

After `pip install .` (or `pip install -e .`), `models` and `src` are regular
importable packages — no `sys.path` manipulation needed:

```python
from models.discontinuous_control import DiscontinuousControl

model = DiscontinuousControl(N=10)

# 1. rejection baseline
import numpy as np
rng = np.random.default_rng(0)
uniform = [x for x in rng.uniform(-2, 2, size=(200, 2)) if model.is_feasible(x)]

# 2. bilevel sampler (needs the ccopt plugin)
bilevel = model.collect_feasible_samples_with_bilevel(
    x_max=np.array([2.0, 2.0]), num_samples=20
)

# 3. linearized Hit-and-Run with nonlinear projection
#    (plot_results defaults to True on this method, so switch it off)
hr = model.collect_feasible_samples_with_hit_and_run(
    num_samples=20, plot_results=False
)
```

---
