import warnings

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import null_space

# Class implementing a model with discontinuous control law
class DiscontinuousControl:
    def __init__(self, N=2, sym_type='SX'):

        self.sym_type = sym_type  # symbolic type for CasADi (SX or MX)
        
        self.N = N
        self.step = self.step_func()  
        self.define_simple_MPC_control(N)  # Define the MPC controller
        
    def step_func(self):
        ''' Define the step function for the system dynamics. '''
        x = ca.SX.sym('x', 2)  # state variable
        u = ca.SX.sym('u', 1)  # control variable
        x_next = x + ca.vertcat(u, u**3)  # system dynamics
        return ca.Function('step', [x, u], [x_next])  # CasADi function for dynamics

    def define_simple_MPC_control(self, N):
        """Define the finite-horizon nonlinear MPC controller."""
        # Define optimization variables and parameters for MPC
        x0 = ca.SX.sym('x0', 2)  # initial state parameter

        X0 = ca.SX.sym('X', 2)  # state trajectory
        U = ca.SX.sym('U', 1, N)    # control trajectory
        decvar = ca.veccat(X0, U)  # decision variables for the optimization problem
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X0, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X0, U], [decvar])
        
        x_max = np.array([4.0, 4.0], dtype=float)
        x_min = -x_max
        # Keep the MPC state bounds available to the sampling routines.
        self.state_bound = x_max.copy()
        
        self.control_bound = 2.0
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        constraints = []  # constraints
        h = []      # equality constraints
        g = []      # inequality constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        
        Xk = ca.SX.sym('X0', 2)  # initial state variable
        w += [Xk]
        self.lbw += list(x_min)  # state bounds
        self.ubw += list(x_max)
        self.w0 += [0.0, 0.0]     # initial guess for state
        constraints += [Xk - x0]  # initial state constraint
        h += [Xk - x0]  # equality constraint for initial state
        g += [Xk - x_max, x_min - Xk]  # inequality constraints for state bounds
        self.lbg += [0.0, 0.0]
        self.ubg += [0.0, 0.0]
        
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += [-self.control_bound]  # control limits
            self.ubw += [self.control_bound]
            self.w0 += [0.0]     # initial guess
            
            # Store inequality constraints for control bounds
            g += [uk - self.control_bound, -uk - self.control_bound]
            
            # Add stage cost to the objective function
            J += ca.sumsqr(Xk) + ca.sumsqr(uk)  

            # Update the state using the system dynamics
            Xk = Xk + ca.vertcat(uk, uk**3)
        
        constraints += [Xk]
        self.lbg += [0.0 for _ in range(2)]
        self.ubg += [0.0 for _ in range(2)]
        
        h += [Xk]
        # Keep the terminal map separate from ``h``: the first block in ``h``
        # depends on the external parameter x0 and is not a terminal constraint.
        w_vec = ca.vertcat(*w)
        self.terminal_constraint = ca.Function(
            'terminal_constraint', [w_vec], [Xk]
        )
        self.jac_terminal_constraint = ca.Function(
            'jac_terminal_constraint',
            [w_vec],
            [ca.jacobian(Xk, w_vec)],
        )
        
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*constraints), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False, }
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
        # Compute the KKT conditions for the MPC problem
        self.f = ca.Function('f', [ca.vertcat(*w)], [J])
        self.grad_f = ca.Function('grad_f', [ca.vertcat(*w)], [ca.gradient(J, ca.vertcat(*w))])
        self.h = ca.Function('h', [x0, ca.vertcat(*w)], [ca.vertcat(*h)])
        self.jac_h = ca.Function('jac_h', [x0, ca.vertcat(*w)], [ca.jacobian(ca.vertcat(*h), ca.vertcat(*w))])
        self.g = ca.Function('g', [ca.vertcat(*w)], [ca.vertcat(*g)])
        self.jac_g = ca.Function('jac_g', [ca.vertcat(*w)], [ca.jacobian(ca.vertcat(*g), ca.vertcat(*w))])
        h_vec = ca.vertcat(*h)
        g_vec = ca.vertcat(*g)
        lam_h = ca.SX.sym('lam_h', h_vec.numel())
        lam_g = ca.SX.sym('lam_g', g_vec.numel())
        lagrangian = J + ca.dot(lam_h, h_vec) + ca.dot(lam_g, g_vec)
        self.L = ca.Function('L', [x0, w_vec, lam_h, lam_g], [lagrangian])
        self.hessian_L = ca.Function(
            'H_L',
            [w_vec, lam_h, lam_g],
            [ca.hessian(lagrangian, w_vec)[0]],
        )
    
    def solve_MPC(self, x0, warm_start=None, ret_seq=False, prev_sol=False, return_traj=False, check_optimality=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        w0 = self.w0 if warm_start is None else warm_start
        sol = self.solver(
            x0=w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        u_opt = sol['x']
        
        if check_optimality and warm_start is not None:
            # Check the distance between the found solution and the warm start.
            w_distance = np.linalg.norm(
                sol['x'].full().flatten() - np.asarray(warm_start, dtype=float).flatten()
            )
            print(f"Trajectory distance: {w_distance:.6f}")
            return bool(w_distance < 5e-4)
        
        if return_traj:
            return sol['x'].full().flatten()  # full MPC decision vector [x0, u_0, ..., u_{N-1}]
        
        if prev_sol:
            self.w0 = sol['x'].full().flatten().tolist()  # warm start for next iteration
        # Return the optimal control sequence
        elif ret_seq:
            return u_opt.full().flatten()  # return the full control sequence as a 1D array
        
        return u_opt.full().flatten()[0]  # return only the first control input
    
    def _linearized_opc_at_solution(
        self,
        solution,
    ):
        """Condense the local constraints to ``G du <= b + F dx0``.

        The nominal decision vector is stacked as
        ``[x0[0], x0[1], u[0], ..., u[N-1]]``.  The terminal equality is
        represented by its two inequality directions.
        """
        if isinstance(solution, dict):
            solution = solution["x"]
        solution = np.asarray(
            solution.full() if hasattr(solution, "full") else solution,
            dtype=float,
        ).reshape(-1)
        if solution.size != self.N + 2:
            raise ValueError(
                f"solution must have {self.N + 2} entries "
                f"([x0(2), U({self.N})]); got {solution.size}."
            )
        if not np.all(np.isfinite(solution)):
            raise ValueError("solution must contain only finite values.")

        x0_opt = solution[:2]
        U_opt = solution[2:]

        # The nonlinear MPC is already in single-shooting form:
        # x_N = x0 + sum_k [u_k, u_k^3].  Hence its terminal Jacobian with
        # respect to U can be assembled directly, without condensing separate
        # per-stage dynamics constraints.
        M = np.vstack((np.ones(self.N), 3.0 * U_opt**2,))

        G_input = np.zeros((2 * self.N, self.N))
        b_input = np.empty(2 * self.N)
        for k, u_nominal in enumerate(U_opt):
            G_input[2 * k:2 * k + 2, k] = (1.0, -1.0)
            b_input[2 * k:2 * k + 2] = (
                self.control_bound - u_nominal,
                self.control_bound + u_nominal,
            )
        F_input = np.zeros((2 * self.N, 2))

        # This linearized OCP has the following structure:
        # M du = -dx0
        # G du <= b

        return {
            "G": G_input,
            "b": b_input,
            "M": M,
            "x0_nominal": x0_opt,
            "U_nominal": U_opt,
        }
        
    def _linearized_boundary_step(
        self,
        dx0,
        direction,
        linear_ocp,
    ):
        """Return the largest feasible nonnegative step along ``direction``.

        This solves

        ``maximize alpha``

        subject to

        ``G du <= b ``, ``M du = -dx0 - alpha*direction`` ``alpha >= 0``.

        Here ``dx0`` is the current perturbation ``dx0_j`` relative to the
        nominal state used to construct ``linear_ocp``.  

        The LP is solved through CasADi's ``conic`` interface with qpOASES.
        """
        G = np.asarray(linear_ocp["G"], dtype=float)
        b = np.asarray(linear_ocp["b"], dtype=float).reshape(-1)
        M = np.asarray(linear_ocp["M"], dtype=float)

        if G.ndim != 2:
            raise ValueError("linear_ocp['G'] must be a matrix.")
        n_constraints, n_du = G.shape
        if b.shape != (n_constraints,):
            raise ValueError(
                "linear_ocp['b'] must contain one entry per row of G."
            )
        if M.ndim != 2 or M.shape[1] != n_du:
            raise ValueError(
                "linear_ocp['M'] must have the same number of columns as G."
            )

        dx0 = np.asarray(dx0, dtype=float).reshape(-1)
        direction = np.asarray(direction, dtype=float).reshape(-1)
        if dx0.shape != (M.shape[0],):
            raise ValueError(
                f"dx0 must have shape ({M.shape[0]},); got {dx0.shape}."
            )
        if direction.shape != dx0.shape:
            raise ValueError(
                f"direction must have shape {dx0.shape}; got {direction.shape}."
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (G, b, M, dx0, direction)
        ):
            raise ValueError("The boundary LP data must contain only finite values.")

        # z = [du, alpha].  
        # The equality constraint is M du + alpha*direction = -dx0
        big_M = np.column_stack((M, direction))
        # Create the A matrix for the inequality constraints: A = [M, direction; G, 0]
        A = np.vstack((big_M, np.column_stack((G, np.zeros(n_constraints)))))
        lba = np.hstack([-dx0, -np.inf * np.ones(n_constraints)])
        uba = np.hstack([-dx0, b])
        # The decision vector is [du, alpha], so the number of variables is n_du + 1.
        n_variables = n_du + 1
        n_equalities = big_M.shape[0]
        n_rows = n_equalities + n_constraints

        cache = getattr(self, "_linearized_boundary_conic_cache", None)
        if cache is None:
            cache = {}
            self._linearized_boundary_conic_cache = cache
        cache_key = (n_constraints, n_variables)
        solver = cache.get(cache_key)
        if solver is None:
            # The LP structure is defined below.
            lp_structure = {
                # A structurally zero Hessian makes this a linear program.
                "h": ca.Sparsity(n_variables, n_variables),
                # This defines the sparsity pattern of the linear constraints.
                # Now the solver knows that there are n_rows x n_variables constraints
                "a": ca.Sparsity.dense(n_rows, n_variables),
            }
            options = {
                "printLevel": "none",
                "print_time": False,
                "error_on_fail": False,
            }
            solver = ca.conic(
                (
                    f"linearized_boundary_{id(self)}_"
                    f"{n_constraints}_{n_variables}"
                ),
                "qpoases",
                lp_structure,
                options,
            )
            cache[cache_key] = solver

        # The objective is to maximize alpha, which is equivalent to minimizing -alpha.
        objective = np.zeros(n_variables)
        objective[-1] = -1.0
        # The decision vector is [du, alpha], so the lower bound on alpha is 0.0 and the upper bound is infinity.
        # du is unbounded
        lbx = np.full(n_variables, -np.inf)
        ubx = np.full(n_variables, np.inf)
        lbx[-1] = 0.0
        try:
            result = solver(
                g=objective,
                a=A,
                lbx=lbx,
                ubx=ubx,
                lba=lba,
                uba=uba,
            )
        except RuntimeError:
            raise RuntimeError("The linearized boundary LP failed to solve.")

        if not solver.stats().get("success", False):
            return None

        solution = result["x"].full().reshape(-1)
        alpha = float(solution[-1])
        if not np.isfinite(alpha):
            return None

        return alpha

    @staticmethod
    def _validate_projection_weight(Qx):
        """Return a symmetric positive-semidefinite 2-by-2 state weight."""
        if Qx is None:
            Qx = np.eye(2)
        elif np.isscalar(Qx):
            Qx = float(Qx) * np.eye(2)
        else:
            Qx = np.asarray(Qx, dtype=float)
            if Qx.shape == (2,):
                Qx = np.diag(Qx)
        Qx = np.asarray(Qx, dtype=float)
        if Qx.shape != (2, 2) or not np.all(np.isfinite(Qx)):
            raise ValueError("Qx must be a finite scalar, length-2 vector, or 2-by-2 matrix.")
        if not np.allclose(Qx, Qx.T, atol=1e-12, rtol=0.0):
            raise ValueError("Qx must be symmetric.")
        if np.min(np.linalg.eigvalsh(Qx)) < -1e-12:
            raise ValueError("Qx must be positive semidefinite.")
        return 0.5 * (Qx + Qx.T)

    def _build_feasible_projection_problem(self):
        """Build the nonlinear projection problem from equation (6).

        With ``w = [x0, U]`` and target ``x0_bar``, the objective is

        ``f(w) + 0.5 * (x0 - x0_bar).T @ Qx @ (x0 - x0_bar)``.

        The single-shooting terminal map supplies ``h_tilde(w) = 0`` and the
        original initial-state and input bounds supply ``g(w) <= 0``.
        """
        sym = getattr(ca, self.sym_type)
        target = sym.sym("projection_target", 2)
        Qx_flat = sym.sym("projection_Qx", 4)
        Qx = ca.reshape(Qx_flat, 2, 2)
        decision = sym.sym("projection_w", self.N + 2)
        x0 = decision[:2]
        displacement = x0 - target

        terminal = self.terminal_constraint(decision)
        inequalities = self.g(decision)
        constraints = ca.vertcat(terminal, inequalities)
        objective = (
            self.f(decision)
            + 0.5 * ca.mtimes([displacement.T, Qx, displacement])
        )
        parameters = ca.vertcat(target, Qx_flat)
        problem = {
            "f": objective,
            "x": decision,
            "g": constraints,
            "p": parameters,
        }
        options = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 2000,
                "tol": 1e-9,
                "acceptable_tol": 1e-7,
            },
        }
        solver = ca.nlpsol(
            f"discontinuous_feasible_projection_{id(self)}",
            "ipopt",
            problem,
            options,
        )
        n_inequalities = int(inequalities.numel())
        return {
            "solver": solver,
            "lbg": np.r_[np.zeros(2), np.full(n_inequalities, -np.inf)],
            "ubg": np.zeros(2 + n_inequalities),
        }

    def project_onto_feasible_set(
        self,
        x_target,
        Qx=None,
        feasible_reference=None,
        reference_solution=None,
        return_solution=False,
    ):
        """Project a target through the original MPC problem's feasible set.

        This is a local nonlinear projection.  A feasible full MPC solution is
        used as the warm start because the terminal equality is non-convex.
        """
        x_target = np.asarray(x_target, dtype=float).reshape(-1)
        if x_target.shape != (2,) or not np.all(np.isfinite(x_target)):
            raise ValueError("x_target must contain two finite values.")
        Qx = self._validate_projection_weight(Qx)

        if reference_solution is None:
            reference = (
                np.zeros(2)
                if feasible_reference is None
                else np.asarray(feasible_reference, dtype=float).reshape(-1)
            )
            if reference.shape != (2,) or not np.all(np.isfinite(reference)):
                raise ValueError("feasible_reference must contain two finite values.")
            try:
                reference_solution = self.solver(
                    x0=self.w0,
                    lbx=self.lbw,
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=reference,
                )
            except Exception as exc:
                raise RuntimeError(
                    "The projection warm-start state is not MPC feasible."
                ) from exc
            if not self.solver.stats().get("success", False):
                raise RuntimeError(
                    "The projection warm-start state is not MPC feasible."
                )

        projection = getattr(self, "_feasible_projection_problem", None)
        if projection is None:
            projection = self._build_feasible_projection_problem()
            self._feasible_projection_problem = projection

        warm_start = (
            reference_solution["x"]
            if isinstance(reference_solution, dict)
            else reference_solution
        )
        parameters = np.r_[x_target, Qx.reshape(-1, order="F")]
        try:
            solution = projection["solver"](
                x0=warm_start,
                lbg=projection["lbg"],
                ubg=projection["ubg"],
                p=parameters,
            )
        except Exception as exc:
            raise RuntimeError("Nonlinear feasible-set projection failed.") from exc
        if not projection["solver"].stats().get("success", False):
            status = projection["solver"].stats().get(
                "return_status", "unknown"
            )
            raise RuntimeError(
                f"Nonlinear feasible-set projection failed ({status})."
            )

        terminal_residual = (
            self.terminal_constraint(solution["x"]).full().reshape(-1)
        )
        inequality_residual = self.g(solution["x"]).full().reshape(-1)
        if (
            np.linalg.norm(terminal_residual, ord=np.inf) > 1e-6
            or np.max(inequality_residual, initial=-np.inf) > 1e-6
        ):
            raise RuntimeError(
                "Nonlinear feasible-set projection returned a constraint-"
                "violating point."
            )

        projected_x0, _ = self.extract_traj(solution["x"])
        projected_x0 = projected_x0.full().reshape(2)
        if return_solution:
            return projected_x0, solution
        return projected_x0

    def plot_linearized_hit_and_run_samples(
        self,
        samples,
        controls=None,
        raw_samples=None,
        seeds=None,
        x_max=None,
        show=True,
        ax=None,
    ):
        """Plot raw and projected Hit-and-Run samples in the state plane."""
        samples = np.asarray(samples, dtype=float).reshape(-1, 2)
        if controls is None:
            controls = np.zeros(samples.shape[0])
        controls = np.asarray(controls, dtype=float).reshape(-1)
        if controls.size != samples.shape[0]:
            raise ValueError("controls must contain one value per sample.")

        if raw_samples is not None:
            raw_samples = np.asarray(raw_samples, dtype=float).reshape(-1, 2)
            if raw_samples.shape[0] != samples.shape[0]:
                raise ValueError(
                    "raw_samples must contain one proposal per projected sample."
                )
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)

        bounds = (
            np.full(2, 4.0)
            if x_max is None
            else np.asarray(x_max, dtype=float).reshape(-1)
        )
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds <= 0.0)
        ):
            raise ValueError("x_max must contain two positive finite values.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure

        if raw_samples is not None and raw_samples.shape[0] > 0:
            ax.scatter(
                raw_samples[:, 0],
                raw_samples[:, 1],
                s=22,
                marker="x",
                color="0.55",
                alpha=0.65,
                label="Linearized HR proposal",
                zorder=1,
            )
            for raw, projected in zip(raw_samples, samples):
                ax.plot(
                    [raw[0], projected[0]],
                    [raw[1], projected[1]],
                    color="0.75",
                    linewidth=0.6,
                    alpha=0.45,
                    zorder=0,
                )

        sample_plot = ax.scatter(
            samples[:, 0],
            samples[:, 1],
            c=controls,
            cmap="coolwarm",
            s=38,
            alpha=0.85,
            edgecolors="none",
            label="Projected feasible sample",
            zorder=2,
        )
        if samples.shape[0] > 0:
            fig.colorbar(sample_plot, ax=ax, label=r"Projected input $u_0^\star$")

        if seeds is not None and seeds.shape[0] > 0:
            ax.scatter(
                seeds[:, 0],
                seeds[:, 1],
                s=110,
                marker="*",
                color="black",
                edgecolors="white",
                linewidths=0.6,
                label="Feasible seed",
                zorder=3,
            )

        ax.add_patch(
            Rectangle(
                (-bounds[0], -bounds[1]),
                2.0 * bounds[0],
                2.0 * bounds[1],
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="0.25",
                label="Sampling bounds",
            )
        )
        ax.set_xlim(-1.05 * bounds[0], 1.05 * bounds[0])
        ax.set_ylim(-1.05 * bounds[1], 1.05 * bounds[1])
        ax.set_xlabel(r"$x_{0,1}$", fontsize=14)
        ax.set_ylabel(r"$x_{0,2}$", fontsize=14)
        # ax.set_title("Linearized Hit-and-Run samples after nonlinear projection")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=12)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def collect_feasible_samples_with_hit_and_run(
        self,
        num_samples=100,
        num_feasible_seeds=5,
        max_seed_attempts=1000,
        # burn_in=10,
        # thinning=1,
        projection_Qx=None,
        random_seed=None,
        x_max=None,
        plot_results=True,
        show=True,
    ):
        """Sample the feasible set with local linearized Hit-and-Run.

        A fixed linearized OCP is constructed around every nonlinear-feasible
        seed.  Each chain evolves in ``dx0`` coordinates relative to that seed.
        Every retained linearized proposal is checked against the original
        nonlinear MPC. Feasible proposals are accepted directly; only
        infeasible proposals are mapped through the nonlinear projection
        problem from equation (6).
        """
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_feasible_seeds <= 0:
            raise ValueError("num_feasible_seeds must be positive.")
        if max_seed_attempts <= 0:
            raise ValueError("max_seed_attempts must be positive.")
        # if burn_in < 0 or thinning <= 0:
        #     raise ValueError(
        #         "burn_in must be non-negative and thinning must be positive."
        #     )
        # The projection weight is validated here, but not used until the nonlinear projection step
        Qx = self._validate_projection_weight(projection_Qx)
        # State bounds definition
        model_bounds = np.full(2, 4.0)
        state_bounds = (model_bounds if x_max is None else np.asarray(x_max, dtype=float).reshape(-1))
        if (
            state_bounds.shape != (2,)
            or not np.all(np.isfinite(state_bounds))
            or np.any(state_bounds <= 0.0)
            or np.any(state_bounds > model_bounds + 1e-12)
        ):
            raise ValueError(
                "x_max must contain two positive finite values no greater "
                "than the MPC state bounds."
            )
        # Check that the number of samples is not zero
        if num_samples == 0:
            self.last_linearized_sampling_stats = {
                "requested_samples": 0,
                "returned_samples": 0,
            }
            return np.empty((0, 2))

        # Generate feasible seeds by sampling the state space and solving the MPC problem
        rng = np.random.default_rng(random_seed)
        seeds = []
        seed_solutions = []
        seed_attempts = 0
        target_seed_count = min(num_feasible_seeds, num_samples)
        while (len(seeds) < target_seed_count and seed_attempts < max_seed_attempts):
            seed_attempts += 1
            candidate = rng.uniform(-state_bounds, state_bounds)
            try:
                solution = self.solver(
                    x0=self.w0,
                    lbx=self.lbw,
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=candidate,
                )
            except Exception:
                continue
            if self.solver.stats().get("success", False):
                solved_x0, _ = self.extract_traj(solution["x"])
                solved_x0 = solved_x0.full().reshape(2)
                terminal_residual = self.terminal_constraint(solution["x"]).full().reshape(-1)
                if (
                    np.any(np.abs(solved_x0) > state_bounds + 1e-7)
                    or np.linalg.norm(
                        solved_x0 - candidate, ord=np.inf
                    ) > 1e-6
                    or np.linalg.norm(
                        terminal_residual, ord=np.inf
                    ) > 1e-6
                ):
                    continue
                # Store the feasible seed and its corresponding optimal solution
                seeds.append(solved_x0)
                seed_solutions.append(solution)
        if not seeds:
            raise RuntimeError(
                f"No feasible seed found after {max_seed_attempts} attempts."
            )

        chains = []
        for seed, solution in zip(seeds, seed_solutions):
            chains.append(
                {
                    "seed": seed.copy(),
                    "seed_solution": solution,
                    "delta_x": np.zeros(2),
                    "linear_ocp": self._linearized_opc_at_solution(solution),
                    "steps": 0,
                }
            )

        samples = []
        controls = []
        raw_candidates = []
        accepted_raw_candidates = []
        raw_candidate_needed_projection = []
        projection_needed = 0
        projection_failures = 0
        boundary_lp_failures = 0
        chain_updates = 0
        chain_index = 0
        max_chain_updates = 10 * num_samples

        while (len(samples) < num_samples and chain_updates < max_chain_updates):
            # Select the next chain in a round-robin fashion
            chain = chains[chain_index % len(chains)]
            chain_index += 1
            # Generate a random direction in the 2D state space
            direction = rng.normal(size=2)
            direction_norm = np.linalg.norm(direction)
            if direction_norm <= np.finfo(float).eps:
                continue
            direction /= direction_norm
            # Compute the maximum feasible step sizes in both directions along the random direction
            alpha_plus = self._linearized_boundary_step(
                chain["delta_x"],
                direction,
                chain["linear_ocp"],
            )
            alpha_minus = self._linearized_boundary_step(
                chain["delta_x"],
                -direction,
                chain["linear_ocp"],
            )
            chain_updates += 1
            if alpha_plus is None or alpha_minus is None:
                boundary_lp_failures += 1
                if alpha_plus is None:
                    print(f"Boundary LP failed for positive direction: {direction}")
                if alpha_minus is None:
                    print(f"Boundary LP failed for negative direction: {direction}")
                continue
            # Sample a step size uniformly from the feasible range and update the chain's state
            beta = rng.uniform(-alpha_minus, alpha_plus)
            chain["delta_x"] += beta * direction
            chain["steps"] += 1
            # # Only retain samples after the burn-in period and according to the thinning parameter
            # if (chain["steps"] <= burn_in or (chain["steps"] - burn_in - 1) % thinning != 0):
            #     continue
            # Compute the candidate state by adding the perturbation to the seed, then project it onto the feasible set
            candidate = chain["seed"] + chain["delta_x"]
            raw_candidates.append(candidate.copy())
            # The linearized OCP only guarantees local linearized feasibility.
            # Classify the raw state against the original nonlinear MPC before
            # projection so sampling comparisons can report how often the
            # linearization itself produced a state requiring correction.
            # candidate_solution = self._feasible_mpc_solution(candidate)
            # needs_projection = candidate_solution is None
            
            # raw_candidate_needed_projection.append(needs_projection)
            # projection_needed += int(needs_projection)
            # if not needs_projection:
            #     _, candidate_U = self.extract_traj(candidate_solution["x"])
            #     samples.append(candidate.copy())
            #     controls.append(float(candidate_U[0]))
            #     accepted_raw_candidates.append(candidate.copy())
            #     continue
            # print(f"Candidate before projection: {candidate}")
            # continue
            try:
                projected, projection_solution = self.project_onto_feasible_set(
                    candidate,
                    Qx=Qx,
                    feasible_reference=chain["seed"],
                    reference_solution=chain["seed_solution"],
                    return_solution=True,
                )
            except RuntimeError:
                projection_failures += 1
                continue
            # Check if projection was needed a posteriori
            # TODO: IMPROVE THIS TEST
            projected_was_needed = not np.allclose(projected, candidate, atol=1e-6, rtol=0.0)
            if projected_was_needed:
                projection_needed += 1
            _, projected_U = self.extract_traj(projection_solution["x"])
            samples.append(projected)
            controls.append(float(projected_U[0]))
            # accepted_raw_candidates.append(candidate.copy())

        result = np.asarray(samples, dtype=float).reshape(-1, 2)
        self.last_feasible_sample_controls = np.asarray(controls, dtype=float)
        self.last_linearized_sampling_stats = {
            "requested_samples": int(num_samples),
            "returned_samples": int(result.shape[0]),
            "seed_attempts": int(seed_attempts),
            "feasible_seeds": np.asarray(seeds, dtype=float).reshape(-1, 2),
            "raw_candidates": np.asarray(raw_candidates, dtype=float).reshape(-1, 2),
            # "accepted_raw_candidates": np.asarray(accepted_raw_candidates, dtype=float).reshape(-1, 2),
            # "raw_candidate_needed_projection": np.asarray(raw_candidate_needed_projection, dtype=bool),
            "projection_needed": int(projection_needed),
            "projection_failures": int(projection_failures),
            "boundary_lp_failures": int(boundary_lp_failures),
            "chain_updates": int(chain_updates),
            # "burn_in": int(burn_in),
            # "thinning": int(thinning),
            "projection_Qx": Qx.copy(),
        }
        if result.shape[0] < num_samples:
            raise RuntimeError(
                f"Collected only {result.shape[0]}/{num_samples} samples "
                f"after {chain_updates} Hit-and-Run updates. Inspect "
                "last_linearized_sampling_stats for diagnostics."
            )
        if plot_results:
            self.last_linearized_sampling_plot = (
                self.plot_linearized_hit_and_run_samples(
                    result,
                    controls=self.last_feasible_sample_controls,
                    raw_samples=accepted_raw_candidates,
                    seeds=seeds,
                    x_max=state_bounds,
                    show=show,
                )
            )
        return result

    def bilevel_active_set(self, x_max, slack_penalty='quadratic', return_solver=False):
        """
        Bilevel active-set method for finding a feasible initial state for the MPC problem.
        By construction, the returned state is guaranteed to be optimally feasible for the MPC problem.
        """
        slack_penalty = str(slack_penalty).lower()
        if slack_penalty not in {"l1", "quadratic"}:
            raise ValueError("slack_penalty must be either 'l1' or 'quadratic'.")

        def weighted_slack_cost(slacks, weights):
            regularization = 1e-6 * ca.dot(slacks, slacks)
            if slack_penalty == "l1":
                return ca.dot(weights, slacks)
            # ca.diag also accepts the symbolic weight parameter used when the
            # solver is built once and the weights are resampled per attempt.
            return 0.5 * slacks.T @ ca.diag(weights) @ slacks
        
        x0 = ca.SX.sym("x0", 2) # initial state parameter
        
        if x_max is None:
            warnings.warn("x_max is None, using default value of 4.0 for both state dimensions.")
            x_max = np.array([4.0, 4.0], dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (2,):
                raise ValueError("x_max must have shape (2,).")
            
        x_min = -x_max
        u_max = np.array([self.control_bound], dtype=float)
        u_min = -u_max    
        
        w = []
        lbw = []
        ubw = []
        w0 = []
        w_mpc = []
        g = []
        lbg = []
        ubg = []
        J = 0
        # Flattened scalar positions of all bound residuals in ca.vertcat(*w).
        # ccopt indexes scalar entries of the final decision vector; it does
        # not index the symbolic blocks stored in the Python list ``w``.
        slack_indices = []
        
        Xk = ca.SX.sym('X0', 2)
        w += [Xk]
        lbw += [-ca.inf] * 2
        ubw += [ca.inf] * 2
        w0 += [0.0] * 2
        w_mpc += [Xk]
        
        s_x0 = ca.SX.sym('slack_x0', 2, 2)
        s_x0_var = ca.vec(s_x0)
        s_x0_start = len(lbw)
        w += [s_x0_var]
        lbw += [0.0 for _ in range(4)]
        ubw += [ca.inf for _ in range(4)]
        w0 += [0.0 for _ in range(4)]
        slack_indices.extend(range(s_x0_start, s_x0_start + s_x0_var.numel()))
        
        g += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
        lbg += [0.0 for _ in range(4)]
        ubg += [0.0 for _ in range(4)]
        
        slacks = [s_x0_var]
        # Each residual block stores upper residuals first, then lower residuals.
        # Keep the block sizes so random spiking can choose at most one side per bound.
        slack_block_sizes = [2]
        
        for k in range(self.N):
            uk = ca.SX.sym(f"u_{k}", 1)
            w += [uk]
            lbw += [-self.control_bound]
            ubw += [self.control_bound]
            w0 += [0.0]
            w_mpc += [uk]
            
            s_uk = ca.SX.sym(f'slack_u_{k}', 1, 2)
            s_uk_var = ca.vec(s_uk)
            s_uk_start = len(lbw)
            w += [s_uk_var]
            lbw += [0.0 for _ in range(2)]
            ubw += [ca.inf for _ in range(2)]
            w0 += [0.0 for _ in range(2)]
            slack_indices.extend(range(s_uk_start, s_uk_start + s_uk_var.numel()))
            slacks.append(s_uk_var)
            slack_block_sizes.append(1)
            
            g += [uk - u_max + s_uk[:, 0], -uk + u_min + s_uk[:, 1]]
            lbg += [0.0 for _ in range(2)]
            ubg += [0.0 for _ in range(2)]
            
            Xk = self.step(Xk, uk)
            
        # Add the terminal constraint
        g += [Xk]
        lbg += [0.0 for _ in range(2)]
        ubg += [0.0 for _ in range(2)]
            
        # Add the slack penalty to the objective function
        rho, _ = self.spiky_residual_weights(
                slack_block_sizes,
                min_spikes=1,
                max_spikes=1,
                base_scale=1e-6,
                spike_factor_range=(1e6, 1e8),
                spike_probability=0.6,
            )
        if return_solver:
            rho_param = ca.SX.sym('rho', int(ca.vertcat(*slacks).numel()))
            J += weighted_slack_cost(ca.vertcat(*slacks), rho_param) + 1e0 * ca.sumsqr(w_mpc[0] - x0)
        else: 
            J += weighted_slack_cost(ca.vertcat(*slacks), rho) + 1e0 * ca.sumsqr(w_mpc[0] - x0)
            
        # Add the stationarity condition of the MPC problem to the constraints
        nabla_f = self.grad_f(ca.vertcat(*w_mpc))
        nabla_h = self.jac_h(w_mpc[0], ca.vertcat(*w_mpc)).T
        nabla_g = self.jac_g(ca.vertcat(*w_mpc)).T
        lam_h = ca.SX.sym('lam_h', nabla_h.shape[1])
        lam_g = ca.SX.sym('lam_g', nabla_g.shape[1])
        stationarity = nabla_f + nabla_h @ lam_h + nabla_g @ lam_g
        w += [lam_h]
        lbw += [-ca.inf] * lam_h.numel()
        ubw += [ca.inf] * lam_h.numel()
        w0 += [0.0] * lam_h.numel()
        # lam_g starts after every primal variable, residual, and equality
        # multiplier already represented by the scalar bound arrays.
        lam_g_start = len(lbw)
        w += [lam_g]
        lbw += [0.0] * lam_g.numel()
        ubw += [ca.inf] * lam_g.numel()
        w0 += [0.0] * lam_g.numel()
        
        g += [stationarity]
        lbg += [0.0] * stationarity.shape[0]
        ubg += [0.0] * stationarity.shape[0]
        
        # Create the NLP problem
        if return_solver:
            params = ca.vertcat(x0, rho_param)
            nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': params}
        else:
            nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p':x0}
            
        varvar = 0

        if len(slack_indices) != lam_g.numel():
            print(f"len(slack_indices) = {len(slack_indices)}, lam_g.numel() = {lam_g.numel()}")
            raise RuntimeError("One residual is required per inequality multiplier.")

        ind_cc = [
            [lam_g_start + i, slack_indices[i]]
            for i in range(lam_g.numel())
        ]
        cctypes = [varvar] * lam_g.numel()
        
        # Create the NLP solver
        print("Creating NLP solver...")
        opts = {
            "expand": False,
            "print_time": False,
            # ccopt delegates NLP solves to MadNLP; bound_relax_factor=0.0 is
            # important for complementarity formulations.
            "cc_pairs": ind_cc,
            "cc_types": cctypes,
            "madnlp.print_level": 6,
            "madnlp.max_iter": 5000,
            "madnlp.tol": 1e-6,
            "madnlp.bound_relax_factor": 0.0,
            "madnlp.linear_solver": "Ma97Solver",
            "ccopt.relaxation_update.TYPE": "RolloffRelaxationUpdate",
            "ccopt.relaxation_update.rolloff_point": 1e-3,
            "ccopt.relaxation_update.rolloff_slope": 1.5,
            "ccopt.relaxation_update.sigma_min": 1e-6,
        }
        bilevel_solver = ca.nlpsol("bilevel_solver", "ccopt", nlp, opts)
        
        mpc_vars = ca.Function('mpc_vars', [ca.vertcat(*w)], [ca.vertcat(*w_mpc)])
        lambda_vars = ca.Function('lambda_vars', [ca.vertcat(*w)], [lam_h, lam_g])
        slack_vars = ca.Function('slack_vars', [ca.vertcat(*w)], [ca.vertcat(*slacks)])
        
        if return_solver:
            return bilevel_solver, w0, lbw, ubw, lbg, ubg, slack_block_sizes, mpc_vars, lambda_vars, slack_vars
        
        # Solve the bilevel problem
        print("Solving bilevel problem...")
        # Sample an initial state
        x0_num = ca.DM([1.0, 0.0])
        sol = bilevel_solver(
            x0=ca.vertcat(*w0),
            lbx=ca.vertcat(*lbw),
            ubx=ca.vertcat(*ubw),
            lbg=ca.vertcat(*lbg),
            ubg=ca.vertcat(*ubg),
            p=x0_num
        )
        return_status = bilevel_solver.stats()['unified_return_status']
        
        if return_status != "SOLVER_RET_SUCCESS":
            raise RuntimeError(f"Bilevel solver failed with status: {return_status}")
        
        x0_feasible = sol['x'][0:2].full().flatten()
        print(f"Feasible initial state found: {x0_feasible}")
        
        # From the optimal solution, extract the mpc solution
        mpc_solution = mpc_vars(sol['x']).full().flatten()
        
        return x0_feasible, mpc_solution
           
    @staticmethod       
    def spiky_residual_weights(
            block_sizes,
            min_spikes=1,
            max_spikes=3,
            base_scale=1.0,
            spike_factor_range=(50, 1000),
            spike_probability=0.4,
        ):
            """Generate residual weights without spiking both sides of the same bound."""
            total_residuals = 2 * sum(block_sizes)
            rho = np.random.exponential(base_scale, size=total_residuals)
    
            if total_residuals <= 0 or np.random.random() > spike_probability:
                return rho, np.array([], dtype=int)
    
            residual_pairs = []
            offset = 0
            for block_size in block_sizes:
                for i in range(block_size):
                    # ca.vec stacks column-major: all upper residuals, then all lower residuals.
                    upper_idx = offset + i
                    lower_idx = offset + block_size + i
                    residual_pairs.append((upper_idx, lower_idx))
                offset += 2 * block_size
    
            max_spikes = min(max_spikes, len(residual_pairs))
            min_spikes = max(1, min(min_spikes, max_spikes))
            if max_spikes < min_spikes:
                return rho, np.array([], dtype=int)
    
            num_spikes = np.random.randint(min_spikes, max_spikes + 1)
            pair_idx = np.random.choice(len(residual_pairs), size=num_spikes, replace=False)
            spike_idx = []
    
            for idx in pair_idx:
                chosen_pair = residual_pairs[idx]
                # Spike one side only, otherwise the optimizer is pushed toward both bounds.
                side = np.random.randint(2)
                spike_idx.append(chosen_pair[side])
    
            spike_idx = np.array(spike_idx, dtype=int)
            factors = np.random.uniform(*spike_factor_range, size=num_spikes)
            rho[spike_idx] *= factors
    
            return rho, spike_idx             

    def seek_x0(
        self,
        N=None,
        x_max=None,
        close_to_bounds=True,
        u0_active_bound=False,
        plot_results=False,
        return_u_seq=False,
        slack_penalty="quadratic",
        print_constraint_jacobian=False,
        return_traj=False,
    ):
        """Find an initial state that yields a feasible optimal solution for the MPC problem.

        The auxiliary NLP replicates the MPC problem (dynamics, input bounds,
        state bounds and the terminal constraint ``x_N = 0``) while replacing
        every bound by a residual slack. Randomly spiked slack weights then pull
        a single bound residual to zero, so repeated calls return different
        states on the boundary of the feasible set.

        Parameters
        ----------
        N : int, optional
            MPC horizon length. Defaults to ``self.N``.
        x_max : array_like, optional
            Maximum absolute values for the state components. If None, defaults
            to the state bounds used by the MPC problem.
        close_to_bounds : bool, optional
            If True, the state residuals are exact (equality) residuals, so the
            spiked weight drives one state to its bound. If False, the residuals
            only over-estimate the distance to the bounds.
        u0_active_bound : bool, optional
            Kept for signature compatibility with the other models; unused.
        plot_results : bool, optional
            If True, plots the resulting trajectory.
        return_u_seq : bool, optional
            If True, returns the full control sequence from the solution.
        return_traj : bool, optional
            If True, returns the solution vector laid out like the MPC decision
            vector ``[x0, u_0, ..., u_{N-1}]`` (usable as a warm start).
        slack_penalty : {"l1", "quadratic"}, optional
            Norm used for the randomly weighted state and input residual slacks.
        print_constraint_jacobian : bool, optional
            If True, checks (and prints) whether the returned state is also
            optimal for the MPC problem via the slack stationarity condition.
        """
        slack_penalty = str(slack_penalty).lower()
        if slack_penalty not in {"l1", "quadratic"}:
            raise ValueError("slack_penalty must be either 'l1' or 'quadratic'.")

        def weighted_slack_cost(slacks, weights):
            regularization = 1e-6 * ca.dot(slacks, slacks)
            if slack_penalty == "l1":
                return ca.dot(ca.DM(weights), slacks)
            return slacks.T @ np.diag(weights) @ slacks

        if N is None:
            N = self.N
        if N < 1:
            raise ValueError("N must be at least 1.")

        if x_max is None:
            x_max = np.asarray(getattr(self, "state_bound", np.array([4.0, 4.0])), dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (2,):
                raise ValueError("x_max must have shape (2,).")
            if np.any(~np.isfinite(x_max)) or np.any(x_max <= 0.0):
                raise ValueError("x_max must contain positive finite values.")

        x_min = -x_max
        u_max = np.array([self.control_bound], dtype=float)
        u_min = -u_max

        # Stage cost of the MPC problem: x'x + u'u, no terminal cost because the
        # terminal state is pinned to the origin by an equality constraint.
        x = ca.SX.sym('x', 2)
        u = ca.SX.sym('u', 1)
        x_next = self.step(x, u)
        l = ca.sumsqr(x) + ca.sumsqr(u)
        F = ca.Function('F', [x, u], [x_next, l])

        w = []
        lbw = []
        ubw = []
        w0 = []
        w_mpc = []
        constraints = []
        lbg = []
        ubg = []
        J = 0

        Xk = ca.SX.sym('X0', 2)
        w += [Xk]
        lbw += [-x_max[0], -x_max[1]]
        ubw += [x_max[0], x_max[1]]
        w0 += [0.0, 0.0]
        w_mpc += [Xk]

        s_x0 = ca.SX.sym('slack_x0', 2, 2)
        s_x0_var = ca.vec(s_x0)
        w += [s_x0_var]
        lbw += [0.0 for _ in range(4)]
        ubw += [ca.inf for _ in range(4)]
        w0 += [0.0 for _ in range(4)]

        if close_to_bounds:
            constraints += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
            lbg += [0.0 for _ in range(4)]
            ubg += [0.0 for _ in range(4)]
        else:
            constraints += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
            lbg += [-ca.inf for _ in range(4)]
            ubg += [0.0 for _ in range(4)]

        X_syms = [Xk]
        U_syms = []
        # x_slacks = [s_x0_var]
        x_slacks = [s_x0_var]
        u_slacks = []
        # Each residual block stores upper residuals first, then lower residuals.
        # Keep the block sizes so random spiking can choose at most one side per bound.
        x_slack_block_sizes = [2]
        u_slack_block_sizes = []

        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            lbw += [-ca.inf]
            ubw += [ca.inf]
            w0 += [0.0]
            w_mpc += [uk]

            s_uk = ca.SX.sym('slack_u_' + str(k), 1, 2)
            s_uk_var = ca.vec(s_uk)
            w += [s_uk_var]
            lbw += [0.0, 0.0]
            ubw += [ca.inf, ca.inf]
            w0 += [0.0, 0.0]

            constraints += [uk - u_max + s_uk[:, 0], -uk + u_min + s_uk[:, 1]]
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

            Xk_next, l_k = F(Xk, uk)

            Xk = ca.SX.sym('X_' + str(k + 1), 2)
            w += [Xk]
            lbw += [-ca.inf, -ca.inf]
            ubw += [ca.inf, ca.inf]
            w0 += [0.0, 0.0]

            # if k < N - 1:
            #     # Only the intermediate states carry bound residuals: the
            #     # terminal state is fixed at the origin by the MPC problem, so
            #     # spiking its residual could not move it anyway.
            #     s_xk = ca.SX.sym('slack_x_' + str(k), 2, 2)
            #     s_xk_var = ca.vec(s_xk)
            #     w += [s_xk_var]
            #     lbw += [0.0 for _ in range(4)]
            #     ubw += [ca.inf for _ in range(4)]
            #     w0 += [0.0 for _ in range(4)]

            #     constraints += [Xk - x_max + s_xk[:, 0], -Xk + x_min + s_xk[:, 1]]
            #     if close_to_bounds:
            #         lbg += [0.0 for _ in range(4)]
            #         ubg += [0.0 for _ in range(4)]
            #     else:
            #         lbg += [-ca.inf for _ in range(4)]
            #         ubg += [0.0 for _ in range(4)]

            #     x_slacks.append(s_xk_var)
            #     # Input has one bounded scalar; state has two bounded scalars.
            #     x_slack_block_sizes.append(2)

            constraints += [Xk_next - Xk]
            lbg += [0.0 for _ in range(2)]
            ubg += [0.0 for _ in range(2)]

            J += l_k
            X_syms.append(Xk)
            U_syms.append(uk)
            u_slacks.append(s_uk_var)
            u_slack_block_sizes.append(1)

        # Terminal constraint of the MPC problem.
        constraints += [Xk]
        lbg += [0.0 for _ in range(2)]
        ubg += [0.0 for _ in range(2)]

        if x_slacks:
            x_slacks_vec = ca.vertcat(*x_slacks)
            if close_to_bounds:
                rho_x, _ = self.spiky_residual_weights(
                    x_slack_block_sizes,
                    min_spikes=1,
                    max_spikes=1,
                    base_scale=1e-6,
                    spike_factor_range=(1e6, 1e8),
                    spike_probability=1.0,
                )
            else:
                rho_x = np.full(x_slacks_vec.numel(), 1e-6, dtype=float)

            J += weighted_slack_cost(x_slacks_vec, rho_x)
        else:
            rho_x = np.zeros(0, dtype=float)

        u_slacks_vec = ca.vertcat(*u_slacks)
        rho_u, _ = self.spiky_residual_weights(
            u_slack_block_sizes,
            min_spikes=1,
            max_spikes=3,
            base_scale=1e-6,
            spike_factor_range=(1e6, 1e8),
            spike_probability=1.0,
        )

        J += weighted_slack_cost(u_slacks_vec, rho_u)

        # Concatenate all slack weights, ordered like ``slack_stack`` below.
        rho = np.concatenate([rho_x, rho_u])

        X_stack = ca.hcat(X_syms)
        U_stack = ca.hcat(U_syms)
        extract = ca.Function("extract_extreme_traj", [ca.vertcat(*w)], [X_stack, U_stack])
        slack_stack = ca.vertcat(*(x_slacks + u_slacks))
        extract_slacks = ca.Function("extract_slacks", [ca.vertcat(*w)], [slack_stack])

        # Every slack is an element of x_dec, so this Jacobian is a selector matrix.
        x_dec = ca.vertcat(*w)
        slack_selector = ca.jacobian(slack_stack, x_dec)
        slack_selector_func = ca.Function(
            "slack_selector",
            [x_dec],
            [slack_selector],
        )

        selector = slack_selector_func(ca.DM.zeros(x_dec.numel())).full()

        # slack_decision_indices[i] gives the position in x_dec of slack_stack[i].
        slack_decision_indices = np.argmax(np.abs(selector), axis=1)

        # Keep a Jacobian for the slacked NLP itself.  Its shape is
        # (constraint rows, decision-variable columns), so a slack decision
        # index must be used as a column index.  Using self.jac_g here would
        # mix the base MPC layout with the interleaved slacked-NLP layout.
        constraint_stack = ca.vertcat(*constraints)
        slacked_constraint_jacobian = ca.Function(
            "slacked_constraint_jacobian",
            [x_dec],
            [ca.jacobian(constraint_stack, x_dec)],
        )

        nlp_prob = {'f': J, 'x': x_dec, 'g': constraint_stack}
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "max_iter": 5000,
                "tol": 1e-8,
            }
        }
        solver = ca.nlpsol("extreme_solver", "ipopt", nlp_prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        return_status = solver.stats()['return_status']

        X_opt, U_opt = extract(sol['x'])
        opt_slacks = extract_slacks(sol['x'])

        # Scalar indices in the interleaved slacked-NLP decision vector.
        all_decision_indices = np.arange(x_dec.numel())
        w_mpc_opt = ca.Function("w_mpc_opt", [x_dec], [ca.vertcat(*w_mpc)])
        w_mpc_solution = w_mpc_opt(sol['x']).full().flatten()

        if print_constraint_jacobian:
            # From opt slacks, extract the weights of the nonzero slacks
            tol = 1e-8

            nonzero_slack_indices = np.flatnonzero(np.abs(opt_slacks) > tol)

            s_I = opt_slacks[nonzero_slack_indices]
            rho_I = rho[nonzero_slack_indices]

            # Compute the Jacobian of the slacked constraints at the optimum.
            # CasADi returns (constraints, decision variables).
            constraint_jacobian_at_opt = (
                slacked_constraint_jacobian(sol['x']).full()
            )

            # Map each active slack decision-variable column to the single
            # constraint row in which that slack occurs.
            constraint_row_I = []

            for slack_idx in nonzero_slack_indices:
                dec_idx = slack_decision_indices[slack_idx]
                rows = np.flatnonzero(
                    np.abs(constraint_jacobian_at_opt[:, dec_idx]) > 1e-12
                )

                if rows.size != 1:
                    raise RuntimeError(
                        f"Slack {slack_idx} appears in constraint rows {rows}"
                    )

                constraint_row_I.append(rows[0])

            constraint_row_I = np.asarray(constraint_row_I, dtype=int)

            # Determine the y variables. They are all variables except x0 and the slacks
            slack_and_x0_indices = np.concatenate([np.arange(2), slack_decision_indices])
            y_decision_indices = np.setdiff1d(all_decision_indices, slack_and_x0_indices)

            # Transpose the selected block so columns correspond to active
            # constraints, as required by the stationarity residual below.
            g_jacobian_y_I = constraint_jacobian_at_opt[
                np.ix_(constraint_row_I, y_decision_indices)
            ].T

            assert g_jacobian_y_I.shape == (y_decision_indices.size, nonzero_slack_indices.size)

            # Check the shapes of the jacobian, rho_i and s_I
            if rho_I.shape[0] != s_I.shape[0]:
                raise RuntimeError(
                    f"rho_I shape {rho_I.shape} does not match s_I shape {s_I.shape}"
                )

            if g_jacobian_y_I.shape[1] != s_I.shape[0]:
                raise RuntimeError(
                    f"g_jacobian_y_I shape {g_jacobian_y_I.shape} does not match s_I shape {s_I.shape}"
                )

            if g_jacobian_y_I.shape[0] != y_decision_indices.shape[0]:
                raise RuntimeError(
                    f"g_jacobian_y_I shape {g_jacobian_y_I.shape} does not match y_decision_indices shape {y_decision_indices.shape}"
                )

            # Build the diagonal matrix of the nonzero slack weights
            Q_I = np.diag(rho_I)

            # Evaluate the following optimality condition
            condition_residual = g_jacobian_y_I @ Q_I @ s_I

            absolute_error = np.linalg.norm(condition_residual)

            scale = (
                np.linalg.norm(g_jacobian_y_I, ord=np.inf)
                * np.linalg.norm(Q_I @ s_I, ord=np.inf)
            )
            relative_error = absolute_error / max(1.0, scale)

            holds = relative_error <= 1e-6

            print("Absolute error:", absolute_error)
            print("Condition holds:", holds)

            # Check if the g_jacobian_y_I matrix is full column rank
            if np.linalg.matrix_rank(g_jacobian_y_I) < g_jacobian_y_I.shape[1]:
                print("g_jacobian_y_I is not full column rank.")

            if np.linalg.matrix_rank(g_jacobian_y_I @ Q_I) < g_jacobian_y_I.shape[1]:
                print("g_jacobian_y_I @ Q_I is not full column rank.")

        x_pred = np.array(X_opt.full())
        u_pred = np.array(U_opt.full()).flatten()

        if plot_results:
            steps_x = np.arange(x_pred.shape[1])
            steps_u = np.arange(u_pred.shape[0])

            fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
            axes[0].plot(steps_x, x_pred[0], "b-o", linewidth=1.5, markersize=3)
            axes[0].axhline(x_max[0], color="k", linestyle="--", alpha=0.4)
            axes[0].axhline(-x_max[0], color="k", linestyle="--", alpha=0.4)
            axes[0].set_ylabel(r"$x_{k,1}$")

            axes[1].plot(steps_x, x_pred[1], "r-o", linewidth=1.5, markersize=3)
            axes[1].axhline(x_max[1], color="k", linestyle="--", alpha=0.4)
            axes[1].axhline(-x_max[1], color="k", linestyle="--", alpha=0.4)
            axes[1].set_ylabel(r"$x_{k,2}$")

            axes[2].step(steps_u, u_pred, where="post", color="k", linewidth=1.8)
            axes[2].set_ylim(-self.control_bound * 1.2, self.control_bound * 1.2)
            axes[2].set_ylabel(r"$u_k$")
            axes[2].set_xlabel(r"$k$")

            for ax in axes:
                ax.grid(True, alpha=0.3)

            fig.suptitle("Extreme x0 Optimization")
            fig.tight_layout()
            plt.show()

        x0_opt = X_opt[:, 0].full().flatten()
        u_seq_opt = U_opt.full().flatten()
        u0_opt = float(u_seq_opt[0])
        if return_u_seq:
            return x0_opt, u0_opt, u_seq_opt, return_status
        elif return_traj:
            return x0_opt, u0_opt, w_mpc_solution, return_status
        return x0_opt, u0_opt, return_status

    def _feasible_mpc_solution(self, x0, tol=1e-6):
        """Return an original-MPC solution for ``x0``, or ``None`` if infeasible."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (2,) or np.any(~np.isfinite(x0)):
            return None

        try:
            sol = self.solver(
                x0=self.w0,
                lbx=self.lbw,
                ubx=self.ubw,
                lbg=self.lbg,
                ubg=self.ubg,
                p=x0,
            )
        except Exception:
            return None

        if not self.solver.stats().get("success", False):
            return None

        solved_x0, solved_u = self.extract_traj(sol["x"])
        solved_x0 = solved_x0.full().reshape(2)
        u_seq = solved_u.full().reshape(self.N)
        terminal_state = solved_x0 + np.array(
            [np.sum(u_seq), np.sum(u_seq**3)]
        )
        feasible = bool(
            np.linalg.norm(solved_x0 - x0, ord=np.inf) <= tol
            and np.linalg.norm(terminal_state, ord=np.inf) <= tol
            and np.all(np.abs(solved_x0) <= 4.0 + tol)
            and np.all(np.abs(u_seq) <= self.control_bound + tol)
        )
        return sol if feasible else None

    def is_feasible(self, x0, tol=1e-6):
        """Return whether the original MPC admits a terminal-feasible solution."""
        return self._feasible_mpc_solution(x0, tol=tol) is not None

    def feasible_x0_bounds_N2(self, x1):
        """Exact bounds on ``x_{0,2}`` of the horizon-2 feasible set, given ``x_{0,1}``.

        With the terminal constraint ``x_2 = 0`` and ``x_{k+1} = x_k + (u_k, u_k^3)``,
        an initial state is feasible iff there exist ``u_0, u_1 in [-U, U]`` with

            u_0 + u_1       = -x_{0,1} =: a
            u_0^3 + u_1^3   = -x_{0,2} =: b

        Writing ``p = u_0 u_1`` and using ``u_0^3 + u_1^3 = a^3 - 3 p a`` gives
        ``p = (a^3 - b) / (3 a)``.  Since ``u_0, u_1`` are the roots of
        ``t^2 - a t + p``, both lie in ``[-U, U]`` iff

            p <= a^2 / 4          (real roots)
            p >= |a| U - U^2      (t^2 - a t + p >= 0 at t = +-U)
            |a| <= 2 U            (vertex inside the interval)

        Substituting ``b = a^3 - 3 a p``, which is monotone in ``p``, and mapping
        back to ``x_0 = (-a, -b)`` yields

            x_{0,1} > 0 :   x_1^3 / 4        <= x_2 <= (x_1 - U)^3 + U^3
            x_{0,1} < 0 :   (x_1 + U)^3 - U^3 <= x_2 <= x_1^3 / 4
            x_{0,1} = 0 :   x_2 = 0

        with ``|x_{0,1}| <= 2U``.  The cubic ``x_2 = x_1^3 / 4`` is the fold of
        ``(u_0, u_1) -> (sum u_k, sum u_k^3)`` (it is attained for ``u_0 = u_1``);
        the other cubic is the bang boundary on which one control saturates.
        The result is intersected with the MPC state box ``|x_0| <= B``.

        Returns ``(lower, upper)`` with ``NaN`` entries where no feasible
        ``x_{0,2}`` exists.
        """
        if self.N != 2:
            raise ValueError("The closed-form feasible set is derived for N = 2 only.")

        x1 = np.asarray(x1, dtype=float)
        U = float(self.control_bound)
        B = np.asarray(getattr(self, "state_bound", np.array([4.0, 4.0])), dtype=float)

        fold = x1**3 / 4.0                 # both controls equal
        bang_upper = (x1 - U)**3 + U**3    # one control at -U, active for x1 >= 0
        bang_lower = (x1 + U)**3 - U**3    # one control at +U, active for x1 <= 0

        lower = np.where(x1 >= 0.0, fold, bang_lower)
        upper = np.where(x1 >= 0.0, bang_upper, fold)

        # The MPC bounds the initial state only; the box therefore just clips the set.
        lower = np.maximum(lower, -B[1])
        upper = np.minimum(upper, B[1])

        infeasible = (np.abs(x1) > min(2.0 * U, B[0])) | (lower > upper)
        lower = np.where(infeasible, np.nan, lower)
        upper = np.where(infeasible, np.nan, upper)
        return lower, upper

    def is_feasible_N2(self, x0, tol=1e-9):
        """Closed-form feasibility test for N = 2 (see :meth:`feasible_x0_bounds_N2`)."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.shape != (2,) or np.any(~np.isfinite(x0)):
            return False
        lower, upper = self.feasible_x0_bounds_N2(x0[0])
        if np.isnan(lower):
            return False
        return bool(lower - tol <= x0[1] <= upper + tol)

    def feasible_x0_region_N2(self, num_points=400):
        """Closed polygons tracing the exact N = 2 feasible set (one per lobe)."""
        U = float(self.control_bound)
        B = np.asarray(getattr(self, "state_bound", np.array([4.0, 4.0])), dtype=float)
        x1_max = min(2.0 * U, B[0])

        polygons = []
        for sign in (1.0, -1.0):
            x1 = sign * np.linspace(0.0, x1_max, int(num_points))
            lower, upper = self.feasible_x0_bounds_N2(x1)
            keep = ~np.isnan(lower)
            if not np.any(keep):
                continue
            x1 = x1[keep]
            lower = lower[keep]
            upper = upper[keep]
            polygons.append(
                np.column_stack(
                    [
                        np.concatenate([x1, x1[::-1]]),
                        np.concatenate([lower, upper[::-1]]),
                    ]
                )
            )
        return polygons

    def plot_feasible_set_N2(self, ax=None, show=True, fill=True, label="exact feasible set (N=2)"):
        """Draw the closed-form feasible initial-state set for N = 2."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 6.0))

        for i, polygon in enumerate(self.feasible_x0_region_N2()):
            if fill:
                ax.fill(
                    polygon[:, 0],
                    polygon[:, 1],
                    color="0.85",
                    zorder=0,
                    label=label if i == 0 else None,
                )
            ax.plot(
                polygon[:, 0],
                polygon[:, 1],
                color="0.35",
                linewidth=1.2,
                zorder=1,
                label=None if (fill or i > 0) else label,
            )

        if show:
            plt.show()
        return ax

    def plot_feasible_samples(
        self,
        samples,
        controls=None,
        x_max=None,
        show=True,
        show_feasible_set=True,
    ):
        """Plot sampled initial states and their associated first controls.

        The left panel shows the sampled feasible set in state space. The right
        panel unwraps the samples by polar angle. For ``N == 2`` the exact
        feasible set derived in :meth:`feasible_x0_bounds_N2` is drawn as a
        backdrop, so the samples can be checked against the true boundary.
        """
        samples = np.asarray(samples, dtype=float)
        if samples.size == 0:
            samples = np.empty((0, 2), dtype=float)
        elif samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must have shape (n_samples, 2).")

        if controls is None:
            controls = np.array([self.solve_MPC(x0) for x0 in samples], dtype=float)
        else:
            controls = np.asarray(controls, dtype=float).reshape(-1)
        if controls.shape[0] != samples.shape[0]:
            raise ValueError("controls must contain one value per sample.")

        if x_max is None:
            x_max = np.ones(2, dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (2,):
                raise ValueError("x_max must have shape (2,).")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        if show_feasible_set and self.N == 2:
            self.plot_feasible_set_N2(ax=axes[0], show=False)
        if samples.shape[0] > 0:
            state_scatter = axes[0].scatter(
                samples[:, 0],
                samples[:, 1],
                c=controls,
                cmap="coolwarm",
                s=38,
                alpha=0.85,
                edgecolors="none",
            )
            fig.colorbar(state_scatter, ax=axes[0], label=r"Feasible input $u_0^\star$")

            theta = np.arctan2(samples[:, 1], samples[:, 0])
            order = np.argsort(theta)
            axes[1].scatter(theta, controls, s=30, color="tab:blue", alpha=0.75)
            axes[1].plot(theta[order], controls[order], color="tab:blue", alpha=0.3)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No feasible samples", ha="center", va="center", transform=ax.transAxes)

        axes[0].add_patch(
            Rectangle(
                (-x_max[0], -x_max[1]),
                2.0 * x_max[0],
                2.0 * x_max[1],
                fill=False,
                linestyle="--",
                linewidth=1.2,
                color="0.25",
                label="sampling bounds",
            )
        )
        axes[0].set_xlim(-1.05 * x_max[0], 1.05 * x_max[0])
        axes[0].set_ylim(-1.05 * x_max[1], 1.05 * x_max[1])
        axes[0].set_xlabel(r"$x_{0,1}$", fontsize=14)
        axes[0].set_ylabel(r"$x_{0,2}$", fontsize=14)
        # axes[0].set_title("Feasible initial states")
        axes[0].legend(loc="best", fontsize=12)

        axes[1].set_xlim(-np.pi, np.pi)
        axes[1].set_xlabel(r"$\theta=\operatorname{atan2}(x_{0,2},x_{0,1})$", fontsize=14)
        axes[1].set_ylabel(r"$u_0^\star$", fontsize=14)
        # axes[1].set_title("Feasible control along the sampled set")
        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def collect_feasible_samples(
        self,
        num_samples=100,
        max_attempts=1000,
        plot_results=True,
        x_max=None,
        return_controls=False,
    ):
        """Collect distinct feasible initial states generated by :meth:`seek_x0`.

        Each successful auxiliary solve already enforces the model dynamics and
        terminal constraint, so its initial state is accepted directly.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative.")
        if x_max is None:
            x_max = np.ones(2, dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (2,):
                raise ValueError("x_max must have shape (2,).")
            if np.any(~np.isfinite(x_max)) or np.any(x_max <= 0.0):
                raise ValueError("x_max must contain positive finite values.")

        samples = []
        controls = []
        successful_statuses = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}

        for attempt in range(max_attempts):
            if len(samples) >= num_samples:
                break

            x0_opt, u0_opt, return_status = self.seek_x0(
                N=self.N,
                x_max=x_max,
                plot_results=False,
            )
            if return_status not in successful_statuses:
                print(
                    f"Attempt {attempt + 1}: feasible-state solve failed "
                    f"(status: {return_status}), skipping..."
                )
                continue
            if any(np.allclose(x0_opt, previous, atol=1e-8, rtol=0.0) for previous in samples):
                print(f"Attempt {attempt + 1}: duplicate sample, skipping...")
                continue

            samples.append(np.asarray(x0_opt, dtype=float))
            controls.append(float(u0_opt))
            print(f"Collected {len(samples)}/{num_samples}")

        samples_array = np.asarray(samples, dtype=float).reshape(-1, 2)
        controls_array = np.asarray(controls, dtype=float)
        self.last_feasible_sample_controls = controls_array

        if len(samples) < num_samples:
            print(f"Only collected {len(samples)}/{num_samples} samples after {max_attempts} attempts")
        else:
            print(f"Collected {num_samples} feasible samples")

        if plot_results:
            self.plot_feasible_samples(
                samples_array,
                controls=controls_array,
                x_max=x_max,
                show=True,
            )

        if return_controls:
            return samples_array, controls_array
        return samples_array
    
    def collect_feasible_samples_with_slacked(
        self,
        x_max=None,
        close_to_bounds=False,
        u0_active_bound=False,
        num_samples=100,
        max_attempts=1000,
        plot_results=True,
        slack_penalty="quadratic",
        print_constraint_jacobian=False,
        print_optimality=False,
        return_controls=False,
    ):
        """Collect initial states found by successful extreme-state solves.

        ``slack_penalty`` selects an L1 or quadratic penalty in the underlying
        extreme-state optimization performed by :meth:`seek_x0`.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative.")
        if x_max is None:
            x_max = np.asarray(getattr(self, "state_bound", np.array([4.0, 4.0])), dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (2,):
                raise ValueError("x_max must have shape (2,).")
            if np.any(~np.isfinite(x_max)) or np.any(x_max <= 0.0):
                raise ValueError("x_max must contain positive finite values.")

        samples = []
        u0_values = []
        attempts_run = 0
        successful_extreme_solves = 0
        feasible_candidates = 0
        successful_statuses = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}

        for attempt in range(max_attempts):
            if len(samples) >= num_samples:
                break
            attempts_run = attempt + 1

            x0_opt, u0_opt, w_opt_slacked, return_status = self.seek_x0(
                self.N,
                x_max,
                close_to_bounds=close_to_bounds,
                u0_active_bound=u0_active_bound,
                plot_results=False,
                slack_penalty=slack_penalty,
                print_constraint_jacobian=print_constraint_jacobian,
                return_traj=True,
            )

            if return_status not in successful_statuses:
                print(f"Attempt {attempt + 1}: solver failed (status: {return_status}), skipping...")
                continue

            if print_optimality:
                # Re-solve the original MPC from the found x0 and compare the
                # solution with the slacked one used as warm start.
                _ = self.solve_MPC(x0_opt, warm_start=w_opt_slacked, return_traj=True, check_optimality=True)

            successful_extreme_solves += 1
            # if not self.is_feasible(x0_opt):
            #     print(f"Attempt {attempt + 1}: x0 not feasible, skipping...")
            #     continue
            feasible_candidates += 1

            samples.append(np.asarray(x0_opt, dtype=float))
            u0_values.append(float(u0_opt))
            print(f"Collected {len(samples)}/{num_samples}")

        if len(samples) < num_samples:
            print(f"Only collected {len(samples)} samples after {max_attempts} attempts")
        else:
            print(f"Collected {num_samples} samples")

        def all_unique(samples):
            return len(samples) == len({
                (x.shape, x.dtype.str, x.tobytes()) for x in samples
            })

        if all_unique(samples):
            print("All collected samples are unique.")
        else:
            print("Warning: Some collected samples are duplicates.")

        x0_samples = np.asarray(samples, dtype=float).reshape(-1, 2)
        u0_values = np.asarray(u0_values, dtype=float)
        self.last_feasible_sample_controls = u0_values
        self.last_feasible_sampling_stats = {
            "slack_penalty": str(slack_penalty).lower(),
            "requested_samples": int(num_samples),
            "returned_samples": int(x0_samples.shape[0]),
            "attempts": int(attempts_run),
            "successful_extreme_solves": int(successful_extreme_solves),
            "feasible_candidates": int(feasible_candidates),
        }

        if plot_results:
            self.plot_feasible_samples(
                x0_samples,
                controls=u0_values,
                x_max=x_max,
                show=True,
            )

        if return_controls:
            return x0_samples, u0_values
        return x0_samples

    def collect_feasible_samples_with_bilevel(self, x_max, num_samples, max_attempts=100, plot_results=False):
        """Collect initial states found by successful solves with bilevel optimization.
        The collected samples are guaranteed to be feasible and optimal for the MPC problem.
        """
        samples = []
        u0_values = []
        attempts_run = 0
        feasible_and_optimal_candidates = 0
        feasible_candidates = 0
        
        # Set up the bilevel optimization problem
        bilevel_solver, w0, lbw, ubw, lbg, ubg, slack_block_sizes, mpc_vars, lambda_vars, slack_vars = self.bilevel_active_set(x_max, return_solver=True)

        for attempt in range(max_attempts):
            if len(samples) >= num_samples:
                break
            attempts_run = attempt + 1
    
            # Sample slack penalty weights for the bilevel optimization
            rho_value, _ = self.spiky_residual_weights(
                        slack_block_sizes,
                        min_spikes=1,
                        max_spikes=2,
                        base_scale=1e-6,
                        spike_factor_range=(1e6, 1e8),
                        spike_probability=0.6,
                    )
            
            x0_value = np.array([0.0, 0.0])
            p_value = np.concatenate([x0_value, rho_value])
            
            # Solve the bilevel optimization problem with the sampled slack penalty weights
            sol = bilevel_solver(
                x0=w0,
                lbx=lbw,
                ubx=ubw,
                lbg=lbg,
                ubg=ubg,
                p=p_value,
            )
            return_status = bilevel_solver.stats()['unified_return_status']
                    
            if return_status != "SOLVER_RET_SUCCESS":
                # raise RuntimeError(f"Bilevel solver failed with status: {return_status}")
                print(f"Attempt {attempt + 1}: solver failed (status: {return_status}), skipping...")
                continue
            
            feasible_candidates += 1
            
            x0_feasible = sol['x'][0:2].full().flatten()
            
            mpc_solution = mpc_vars(sol['x']).full().flatten()
            u0_feasible = mpc_solution[2:3]
            lambda_opt, mu_opt = lambda_vars(sol['x'])
            slack_opt = slack_vars(sol['x'])
            
            # The reduced Hessian is defined on the null space of the
            # equality and active-inequality constraint Jacobians. A zero
            # slack identifies an active original inequality.
            active_mask = slack_opt.full().flatten() < 1e-6
            jac_h = self.jac_h(x0_feasible, mpc_solution).full()
            jac_g = self.jac_g(mpc_solution).full()
            if active_mask.size != jac_g.shape[0]:
                raise RuntimeError(
                    "The inequality slack vector and constraint Jacobian "
                    f"have incompatible sizes ({active_mask.size} and "
                    f"{jac_g.shape[0]})."
                )
            active_constraint_jacobian = np.vstack(
                [jac_h, jac_g[active_mask, :]]
            )
            Z = null_space(active_constraint_jacobian)
            
            # Compute the reduced Hessian of the Lagrangian at the solution
            H = self.hessian_L(mpc_solution, lambda_opt, mu_opt).full()
            H_reduced = Z.T @ H @ Z
            
            # Apply the Cholesky decomposition to check if the reduced Hessian is positive definite
            try:
                np.linalg.cholesky(H_reduced)
                kkt_residual = 0.0
                print("Reduced Hessian is positive definite.")
            except np.linalg.LinAlgError:
                kkt_residual = np.linalg.norm(H_reduced - H_reduced.T) + np.sum(np.minimum(np.linalg.eigvalsh(H_reduced), 0.0)**2)
                print(f"Reduced Hessian is not positive definite. KKT residual: {kkt_residual:.6e}")
            
            optimality = True if kkt_residual < 5e-4 else False

            # optimality = optimality and self.solve_MPC(
            #     x0_feasible,
            #     warm_start=mpc_solution,
            #     evaluate_kkt=True,
            #     check_optimality=True,
            # )
            
            if optimality:
                feasible_and_optimal_candidates += 1
                samples.append(x0_feasible)
                u0_values.append(u0_feasible)
            print(f"Collected {len(samples)}/{num_samples}")

        if len(samples) < num_samples:
            print(f"Only collected {len(samples)} samples after {max_attempts} attempts")
        else:
            print(f"Collected {num_samples} samples")
        
        x0_samples = np.array(samples, dtype=float)
        u0_values = np.array(u0_values, dtype=float)
        self.last_feasible_sample_controls = u0_values
        self.last_feasible_sampling_stats = {
            "requested_samples": int(num_samples),
            "returned_samples": int(x0_samples.shape[0]),
            "attempts": int(attempts_run),
            "feasible_and_optimal_candidates": int(feasible_and_optimal_candidates),
            "feasible_candidates": int(feasible_candidates),
        }
        
        # Print statistics about the collected samples
        print("\n")
        print(f"Feasible and optimal candidates over feasible candidates: {feasible_and_optimal_candidates}/{feasible_candidates} ({feasible_and_optimal_candidates/feasible_candidates*100:.2f}%)")
        print(f"Feasible candidates over attempts: {feasible_candidates}/{attempts_run} ({feasible_candidates/attempts_run*100:.2f}%)")
        print(f"Total attempts: {attempts_run}")
        
        if plot_results:
            self.plot_feasible_samples(
                x0_samples,
                controls=u0_values,
                x_max=x_max,
                show=True,
            )
        
        return x0_samples
 
    def close_loop_simulation(self, x0, Nsim = 60, control_policy=None, plot_results=True):
        """Simulate the system and plot the results.
        
        Parameters
        ----------
        x0 : array_like
            Initial state.
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step. If None, the MPC controller will be used.
        Nsim : int, optional
            Number of simulation steps (default: 60).
        plot_results : bool, optional
            Whether to plot the results (default: True).
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (Nsim+1, 2).
        u_traj : np.ndarray
            Control trajectory of shape (Nsim,).
        """
        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, 2))
        u_traj = np.zeros(Nsim)
        
        x_traj[0] = np.array(x0).flatten()
        
        # Simulate forward in time
        for k in range(Nsim):
            # Get control input
            if control_policy is None:
                u_k = self.solve_MPC(x_traj[k])  # Solve MPC to get control input for current state
            elif callable(control_policy):
                u_k = control_policy(x_traj[k])  # Get control input from the provided policy function
            else:
                u_k = control_policy[k]  # Get control input from the provided array
            
            # Ensure u_k is scalar
            if hasattr(u_k, '__len__'):
                u_k = float(u_k[0])
            else:
                u_k = float(u_k)
            
            u_traj[k] = u_k
            
            # Step forward
            x_next = self.step(x_traj[k], [u_k])
            x_traj[k + 1] = np.array(x_next.full()).flatten()
        
        if not plot_results:
            return x_traj, u_traj
        
        # Plot results
        time = np.arange(Nsim + 1) * self.dt
        time_u = np.arange(Nsim) * self.dt
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        title = "System Simulation"
        
        # X1
        axes[0].plot(time, x_traj[:, 0], 'b-', linewidth=2)
        axes[0].set_ylabel(r"Position $x_1$ (m)", fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        
        # X2
        axes[1].plot(time, x_traj[:, 1], 'r-', linewidth=2)
        axes[1].set_ylabel(r"Velocity $x_2$ (m/s)", fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Control
        axes[2].step(time_u, u_traj, 'k-', where='post', linewidth=2)
        axes[2].set_ylabel(r"Control $u$ (N)", fontsize=10)
        axes[2].set_xlabel(r"Time $t$ (s)", fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj
    
    def find_first_control(self, N=100):
        """the first control input from the MPC policy as a function of the initial state."""
        # Sample the initial state space in the unit circle
        r = 1.0
        theta = np.linspace(-np.pi, np.pi, N)
        x1_vals = r * np.cos(theta)
        x2_vals = r * np.sin(theta)
        U = np.zeros(N)
        
        initial_states = []
        for i in range(N):
            x0 = [x1_vals[i], x2_vals[i]]
            U[i] = self.solve_MPC(x0, prev_sol=True)  # warm start for faster convergence
            initial_states.append(x0)
        
        return initial_states, U    
    
    def plot_first_control(self, N=100):
        """Plot the first control input from the MPC policy as a function of the initial state."""
        initial_states, U = self.find_first_control(N)
        theta = np.array([np.arctan2(x0[1], x0[0]) for x0 in initial_states])  # angle of the initial state in polar coordinates
        
        # Plot the control input as a function of theta
        plt.figure(figsize=(8, 6))
        plt.scatter(theta, U.flatten(), s=4, alpha=0.5)
        # plt.plot(theta, U.flatten(), c='b', alpha=0.5)
        plt.xlabel(r"$\theta$ (rad)", fontsize=10)
        plt.ylabel(r"$u_0^\star$ (N)", fontsize=10)
        plt.title(r"First MPC input $u_0^\star(\theta)$", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.show()
    

if __name__ == "__main__":
    # Create cart-pole environment
    sys = DiscontinuousControl()
    
    # Example usage: simulate with zero control input
    state_bound = 1.0
    x1 = np.random.uniform(-state_bound, state_bound)
    x2 = np.random.uniform(-state_bound, state_bound)
    x0 = [x1, x2]
    # print (f"Initial state: {x0}")
    
    # sys.close_loop_simulation(x0, Nsim=3)
    # sys.plot_first_control(N=300)
    # sys.collect_feasible_samples_with_slacked(
    #     x_max=[4.0, 4.0],
    #     close_to_bounds=True,
    #     u0_active_bound=False,
    #     num_samples=1000,
    #     max_attempts=10000,
    #     plot_results=True,
    #     slack_penalty="quadratic",
    #     print_constraint_jacobian=False,
    #     print_optimality=True,
    #     return_controls=False
    # )
    
    sys.collect_feasible_samples_with_bilevel(
        x_max=[4.0, 4.0],
        num_samples=1000,
        max_attempts=10000,
        plot_results=True,
    )
    # sys.collect_feasible_samples_with_hit_and_run(
    #     num_samples=1000,
    #     num_feasible_seeds=5,
    #     projection_Qx=100.0,
    #     random_seed=42,
    #     plot_results=True
    # )
