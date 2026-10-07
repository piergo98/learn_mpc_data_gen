# Computationally Efficient Data Generation for Learning Nonlinear MPC Policies

Code for generating feasible, locally optimal initial states for learning NMPC
policies. It combines active-set exploration (bilevel / slacked problems) with
projection-based Hit-and-Run sampling.

- `models/` — the dynamical systems and their samplers
- `src/` — the comparison scripts
- `results/` — output of the scripts

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install .            # or: pip install -e .
```

### CasADi with `ccopt`

The bilevel sampler uses the `ccopt` complementarity solver, which is not in
the PyPI CasADi wheel. Build CasADi with `ccopt` following
[CasADi repo](https://github.com/casadi/casadi), then point
Python to it:

```bash
export PYTHONPATH=/path/to/casadi-ccopt/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/casadi-ccopt/lib:$LD_LIBRARY_PATH
```

## Running the comparisons

Run from the repository root. Each script writes CSV summaries and figures to
`results/` (change with `--output-dir`); add `--show` to display the plots and
`--help` to list all options.

### Discontinuous control

```bash
python3 src/discontinuous_control_projection_sampling_comparison.py \
    --num-samples 100 --num-references 10 --horizon 2 --repeats 3
```

### Cart-pole, horizon sweep

```bash
python3 src/cart_pole_horizon_sampling_comparison.py \
    --horizons 10 20 30 --num-samples 50 --num-references 5 --repeats 3
```

### Main arguments

| Argument | Meaning |
| --- | --- |
| `--num-samples` | Number of initial states generated per method |
| `--num-references` | Bilevel/slacked references `Np`; each yields `num-samples / Np` Hit-and-Run candidates |
| `--horizon` / `--horizons` | MPC horizon (cart-pole: list of horizons to sweep) |
| `--repeats` | Independent reruns used for mean ± std of timing and acceptance |
| `--seed` | Random seed |
| `--projection-weight` | Weight of the state distance in the nonlinear projection |
| `--warm-start / --no-warm-start` | Warm-start consecutive bilevel solves and projections |
| `--rollout-steps`, `--rollout-seeds` | Closed-loop rollout length and number of seeds (0 = automatic) |
| `--max-attempts` | Maximum candidates for uniform rejection sampling |
| `--verbose` | Show solver output |
| `--output-dir` | Where results are saved |
