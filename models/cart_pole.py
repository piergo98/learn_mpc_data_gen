import time
import warnings

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are, null_space

try:
    from .scholtes_homotopy import ScholtesHomotopySolver
except ImportError:  # Support running this file directly from the repository.
    from scholtes_homotopy import ScholtesHomotopySolver


# IPOPT and LP solvers can return nominal points a few ulps outside an active
# bound. Use one shared tolerance when transferring those points to the
# condensed Hit-and-Run LP.
BOUNDARY_LP_FEASIBILITY_TOL = 1e-7


# Class implementing the cart-pole environment
class CartPole:
    def __init__(self, N, dt=0.05, dt_mpc=None, dt_sim=None, sym_type='SX'):
        self.dt_mpc = dt if dt_mpc is None else dt_mpc
        self.dt_sim = self.dt_mpc if dt_sim is None else dt_sim
        self.dt = self.dt_mpc  # Backward-compatible alias for MPC time step.
        self.g = 9.81  # gravity
        self.l = 0.5   # length of the pole
        self.m = 0.08   # mass of the pole
        self.M = 0.8   # mass of the cart
        self.sym_type = sym_type  # symbolic type for CasADi (SX or MX)
        
        self.dynamics = self.dynamics_func()  # CasADi function for dynamics
        self.mpc_step = self.step_func(self.dt_mpc)  # RK4 step used by MPC prediction
        self.sim_step = self.step_func(self.dt_sim)  # RK4 step used by closed-loop simulation
        self.step = self.mpc_step  # Backward-compatible alias for existing callers.
        self.lin_dyn = self.linearized_dynamics()  # CasADi function for linearized dynamics
        self.discrete_lin_dyn = self.discrete_linearized_dynamics()
        
        self.N = N  # MPC horizon
        # Populated by seek_x0() with the optimized decision vector after
        # removing every slack-variable component.
        self.seek_x0_opt_without_slacks = None
        # self.Q = ca.diag(ca.DM([2.5, 0.25, 2.5, 0.25]))  # state cost weights
        # self.R = ca.diag(ca.DM([0.01]))  # control cost weight
        self.Q = ca.diag(ca.DM([2.5, 0.01, 10.0, 0.01]))  # state cost weights
        self.R = ca.diag(ca.DM([0.1]))  # control cost weight
        self.define_simple_MPC_control(self.N)  # Define the MPC controller

    def discrete_linearized_dynamics(self):
        """Return the Jacobians of the RK4 MPC dynamics.
        
        x[k+1] = f(x[k], u[k]) is approximated as:
        dx[k+1] ~= A[k] dx[k] + B[k] du[k]
        where dx = x - x_eq and du = u - u_eq.
        
        """
        x = getattr(ca, self.sym_type).sym('x_discrete_lin', 4)
        u = getattr(ca, self.sym_type).sym('u_discrete_lin', 1)
        x_next = self.mpc_step(x, u)
        A = ca.jacobian(x_next, x)
        B = ca.jacobian(x_next, u)
        return ca.Function('discrete_linearized_dynamics', [x, u], [A, B])

    def dynamics_func(self):
        ''' Computes the time derivative of the state given the current state and control input. '''
        # x = [position, velocity, angle, angular_velocity]
        # u = [force]
        
        x = getattr(ca, self.sym_type).sym('x', 4)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[2])
        c = ca.cos(x[2])
        
        x_dot = x[1]
        theta_dot = x[3]
        
        common_denominator = self.M + self.m * s**2
        
        x_ddot = (u[0] + self.m * s * (theta_dot**2 * self.l - self.g * c)) / common_denominator
        theta_ddot = (-u[0] * c - self.m * s * c * theta_dot**2 * self.l + (self.M + self.m) * self.g * s) / (self.l * common_denominator)
        
        return ca.Function('dynamics', [x, u], [ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)])
    
    def step_func(self, dt):
        ''' Computes the next state given the current state and control input using RK4 integration. '''
        x0 = getattr(ca, self.sym_type).sym('x0', 4)
        u0 = getattr(ca, self.sym_type).sym('u0', 1)
        M = 4  # number of stages for RK4
        DT = dt / M
        X = x0
        for i in range(M):
            k1 = self.dynamics(X, u0)
            k2 = self.dynamics(X + DT/2 * k1, u0)
            k3 = self.dynamics(X + DT/2 * k2, u0)
            k4 = self.dynamics(X + DT * k3, u0)
            X = X + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
        x_next = X
        
        return ca.Function('step', [x0, u0], [x_next])
    
    def linearized_dynamics(self):
        ''' Computes the linearized dynamics matrices A and B around an equilibrium point (x_eq, u_eq). '''
        x = getattr(ca, self.sym_type).sym('x', 4)  # state vector
        u = getattr(ca, self.sym_type).sym('u', 1)  # control input
        
        s = ca.sin(x[2])
        c = ca.cos(x[2])
        
        x_dot = x[1]
        theta_dot = x[3]
        
        common_denominator = self.M + self.m - self.m * c**2
        
        x_ddot = (u[0] + self.m * s * (theta_dot**2 * self.l - self.g * c)) / common_denominator
        theta_ddot = (-u[0] * c - self.m * s * c * theta_dot**2 * self.l + (self.M + self.m) * self.g * s) / (self.l * common_denominator)
        A_sym = ca.jacobian(ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot), x)
        B_sym = ca.jacobian(ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot), u)
        
        return ca.Function('linearized_dynamics', [x, u], [A_sym, B_sym])
    
    def define_simple_MPC_control(self, N):
        ''' A simple MPC controller that tries to stabilize the pole at the upright position. '''
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 4)  # state variable
        u = ca.SX.sym('u', 1)  # control variable
        x0 = ca.SX.sym('x0', 4)  # initial state parameter
        
        X = ca.SX.sym('X', 4, N+1)  # state trajectory
        U = ca.SX.sym('U', 1, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Define cost function (quadratic cost on state deviation and control effort)
        xr = ca.DM([0.0, 0, 0, 0])  # desired state (upright position)
        
        A, B = self.lin_dyn(xr, ca.DM([0]))  # linearized dynamics around the upright position
        E = solve_continuous_are(A.full(), B.full(), self.Q.full(), self.R.full())
        
        x_next = self.mpc_step(x, u)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        # self.p_bound = 2.0
        # self.v_bound = 4.0
        # self.theta_bound = np.pi/3
        # self.omega_bound = 2.0
        
        self.p_bound = 2.0
        self.v_bound = 5.0
        self.theta_bound = np.pi
        self.omega_bound = 10.0
        
        self.x_max = np.array([self.p_bound, self.v_bound, self.theta_bound, self.omega_bound], dtype=float)
        self.x_min = -self.x_max
        
        self.u_bound = 10.0
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        constraints = []      # constraints
        h = []      # equality constraints
        g = []      # inequality constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', 4)  # initial state variable
        w += [Xk]
        self.lbw += [-self.p_bound, -self.v_bound, -self.theta_bound, -self.omega_bound]
        self.ubw += [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound]
        self.w0 += [0.0 for _ in range(4)]
        constraints += [Xk - x0]  # initial state constraint
        # Store equality and inequality constraints separately for KKT evaluation
        h += [Xk - x0]
        g += [Xk - self.x_max, -Xk + self.x_min]  # state bounds 
        self.lbg += [0.0 for _ in range(4)]
        self.ubg += [0.0 for _ in range(4)]
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            self.lbw += [-self.u_bound]  # control limits
            self.ubw += [self.u_bound]
            self.w0 += [0.0]     # initial guess
            # Store inequality constraints separately for KKT evaluation
            g += [uk - self.u_bound, -uk + (-self.u_bound)]  # control bounds
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += [-self.p_bound, -self.v_bound, -self.theta_bound, -self.omega_bound]
            self.ubw += [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound]
            self.w0 += [0.0 for _ in range(4)]
            
            # Store inequality constraints separately for KKT evaluation
            g += [Xk - self.x_max, -Xk + self.x_min]  # state bounds
            
            constraints += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            
            # Store equality constraints separately for KKT evaluation
            h += [Xk_next - Xk]
            
            J += l_k  # accumulate cost
        J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)  # terminal cost
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*constraints), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False, "verbose": False}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        # The projection problem has the same horizon and must be rebuilt if
        # callers redefine the MPC with a different horizon.
        self._feasible_projection_problem = None
        self._linearized_boundary_solvers = {}
        
        # Compute the KKT conditions for the MPC problem
        self.f = ca.Function('f', [ca.vertcat(*w)], [J])
        self.grad_f = ca.Function('grad_f', [ca.vertcat(*w)], [ca.gradient(J, ca.vertcat(*w))])
        self.h = ca.Function('h', [x0, ca.vertcat(*w)], [ca.vertcat(*h)])
        self.jac_h = ca.Function('jac_h', [x0, ca.vertcat(*w)], [ca.jacobian(ca.vertcat(*h), ca.vertcat(*w))])
        self.g = ca.Function('g', [ca.vertcat(*w)], [ca.vertcat(*g)])
        self.jac_g = ca.Function('jac_g', [ca.vertcat(*w)], [ca.jacobian(ca.vertcat(*g), ca.vertcat(*w))])
        w_vec = ca.vertcat(*w)
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
         
    def solve_MPC(self, x0, warm_start=None, ret_seq=False, return_traj=False, evaluate_kkt=False, check_optimality=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        if warm_start is not None:
            w0 = warm_start
        else:
            w0 = self.w0
        sol = self.solver(
            x0=w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        x_opt, u_opt = self.extract_traj(sol['x'])
        
        if evaluate_kkt:
            # The NLP passed to IPOPT contains only the initial-state and
            # dynamics equalities in its constraint vector. Consequently,
            # sol['lam_g'] multiplies h, while sol['lam_x'] contains the
            # multipliers associated with the state and input variable bounds.
            print(self.solver.stats()['return_status'])
            grad_f_val = self.grad_f(sol['x']).full().flatten()
            jac_h_val = self.jac_h(x0, sol['x']).full()
            lam_h_val = sol['lam_g'].full().flatten()
            lam_x_val = sol['lam_x'].full().flatten()
            
            stationarity = grad_f_val + jac_h_val.T @ lam_h_val + lam_x_val
            kkt_residual = np.linalg.norm(stationarity)
            print(f"KKT stationarity residual norm: {kkt_residual:.6e}")
        
        if check_optimality and warm_start is not None:
            # Check the distance between the found solution and the warm start (if provided)
            w_distance = np.linalg.norm(sol['x'].full().flatten() - warm_start)
            print(f"Trajectory distance: {w_distance:.6f}")
            return True if w_distance < 5e-4 else False
        
        # Return the optimal control sequence
        if ret_seq:
            return u_opt.full().flatten()  # return the full control sequence as a 1D array
        
        if return_traj:
            return sol['x'].full().flatten()  # return the full state and control trajectories

        return u_opt.full().flatten()[0]  # return only the first control input

    @staticmethod
    def _state_component_index(component):
        """Map a state component name or index to its integer index."""
        component_names = {
            "position": 0,
            "pos": 0,
            "x": 0,
            "velocity": 1,
            "vel": 1,
            "v": 1,
            "angle": 2,
            "theta": 2,
            "angular_velocity": 3,
            "omega": 3,
            "theta_dot": 3,
        }
        if isinstance(component, str):
            key = component.lower()
            if key not in component_names:
                raise ValueError(f"Unknown state component '{component}'.")
            return component_names[key]

        idx = int(component)
        if idx < 0 or idx >= 4:
            raise ValueError("State component indices must be in [0, 3].")
        return idx

    @staticmethod
    def _normalize_state_batch(x0_batch):
        """Return a batch of states with shape (n_batch, 4)."""
        x0_arr = np.asarray(x0_batch, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        elif x0_arr.ndim != 2:
            raise ValueError("x0_batch must have shape (n_batch, 4) or (4, n_batch).")

        if x0_arr.shape[1] != 4 and x0_arr.shape[0] == 4:
            x0_arr = x0_arr.T
        if x0_arr.shape[1] != 4:
            raise ValueError("Each initial state must have length 4.")
        return x0_arr

    def solve_MPC_batch_first_controls(
        self,
        x0_batch,
        state_components=(0, 1, 2, 3),
        x_axis_component=0,
        plot_results=True,
        compare_uniform=False,
        uniform_seed=None,
        sort_by_x_axis=True,
        show=True,
        skip_failed=False,
    ):
        """Compute and optionally plot the first MPC action for a batch of states.

        Parameters
        ----------
        x0_batch : array_like
            Initial states with shape ``(n_batch, 4)`` or ``(4, n_batch)``.
        state_components : sequence of int or str
            Initial-state components to plot. Names such as ``"position"``,
            ``"velocity"``, ``"theta"``, and ``"omega"`` are accepted.
        x_axis_component : int or str
            State component used as the common plot x-axis.
        plot_results : bool, optional
            Whether to create the plot.
        compare_uniform : bool, optional
            Whether to solve and display a uniformly sampled comparison batch
            with the same number of samples.
        uniform_seed : int, optional
            Seed used only for the uniform comparison batch.
        sort_by_x_axis : bool, optional
            If True, sort samples by the x-axis component before plotting.
        show : bool, optional
            Whether to call ``plt.show()``.
        skip_failed : bool, optional
            If True, failed MPC solves are kept as NaN controls. If False, the
            original exception is raised.

        Returns
        -------
        u0_values : np.ndarray
            First MPC controls with shape ``(n_batch,)`` in the input order.
        fig, axes : tuple or (None, None)
            Matplotlib figure and axes when ``plot_results=True``; otherwise
            ``(None, None)``.
        """
        x0_arr = self._normalize_state_batch(x0_batch)
        component_indices = [self._state_component_index(comp) for comp in state_components]
        x_axis_idx = self._state_component_index(x_axis_component)

        u0_values = np.full(x0_arr.shape[0], np.nan, dtype=float)
        for i, x0 in enumerate(x0_arr):
            try:
                u0_values[i] = float(self.solve_MPC(x0, ret_seq=False))
            except Exception:
                if not skip_failed:
                    raise
                print(f"Skipping first-control solve for state {i + 1}; storing NaN.")

        if not plot_results:
            return u0_values, None, None

        uniform_x0_arr = None
        uniform_u0_values = None
        if compare_uniform:
            rng = np.random.default_rng(uniform_seed)
            bounds = np.array(
                [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound],
                dtype=float,
            )
            uniform_x0_arr = rng.uniform(
                low=-bounds,
                high=bounds,
                size=x0_arr.shape,
            )
            uniform_u0_values = np.full(uniform_x0_arr.shape[0], np.nan, dtype=float)
            for i, x0_uniform in enumerate(uniform_x0_arr):
                try:
                    uniform_u0_values[i] = float(self.solve_MPC(x0_uniform, ret_seq=False))
                except Exception:
                    if not skip_failed:
                        raise
                    print(f"Skipping uniform first-control solve for state {i + 1}; storing NaN.")

            self.uniform_comparison_x0 = uniform_x0_arr
            self.uniform_comparison_u0 = uniform_u0_values

        fig, axes = self.plot_initial_state_controls(
            x0_arr,
            u0_values,
            state_components=component_indices,
            x_axis_component=x_axis_idx,
            comparison_x0_batch=uniform_x0_arr,
            comparison_u0_values=uniform_u0_values,
            comparison_label="uniform",
            sort_by_x_axis=sort_by_x_axis,
            show=show,
        )
        return u0_values, fig, axes

    def plot_initial_state_controls(
        self,
        x0_batch,
        u0_values,
        state_components=(0, 1, 2, 3),
        x_axis_component=0,
        comparison_x0_batch=None,
        comparison_u0_values=None,
        comparison_label="comparison",
        sort_by_x_axis=True,
        show=True,
    ):
        """Plot selected initial-state components and matching first controls."""
        x0_arr = self._normalize_state_batch(x0_batch)
        u0_values = np.asarray(u0_values, dtype=float).reshape(-1)
        if u0_values.shape[0] != x0_arr.shape[0]:
            raise ValueError("u0_values length must match the number of initial states.")

        component_indices = [self._state_component_index(comp) for comp in state_components]
        x_axis_idx = self._state_component_index(x_axis_component)

        order = np.arange(x0_arr.shape[0])
        if sort_by_x_axis:
            order = np.argsort(x0_arr[:, x_axis_idx])

        x_plot = x0_arr[order, x_axis_idx]
        x0_plot = x0_arr[order]
        u0_plot = u0_values[order]
        valid = np.isfinite(u0_plot)

        comparison_x_plot = None
        comparison_x0_plot = None
        comparison_u0_plot = None
        comparison_valid = None
        if comparison_x0_batch is not None:
            comparison_x0_arr = self._normalize_state_batch(comparison_x0_batch)
            comparison_u0_values = np.asarray(comparison_u0_values, dtype=float).reshape(-1)
            if comparison_u0_values.shape[0] != comparison_x0_arr.shape[0]:
                raise ValueError("comparison_u0_values length must match the comparison initial states.")

            comparison_order = np.arange(comparison_x0_arr.shape[0])
            if sort_by_x_axis:
                comparison_order = np.argsort(comparison_x0_arr[:, x_axis_idx])
            comparison_x_plot = comparison_x0_arr[comparison_order, x_axis_idx]
            comparison_x0_plot = comparison_x0_arr[comparison_order]
            comparison_u0_plot = comparison_u0_values[comparison_order]
            comparison_valid = np.isfinite(comparison_u0_plot)

        state_labels = [
            "Position (m)",
            "Velocity (m/s)",
            "Angle (rad)",
            "Angular Velocity (rad/s)",
        ]
        colors = ["tab:blue", "tab:red", "tab:green", "tab:cyan"]

        n_rows = len(component_indices) + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.3 * n_rows), sharex=True)
        axes = np.atleast_1d(axes)

        hist_ax = axes[0].twinx()
        hist_ax.hist(x_plot, bins="auto", color="tab:purple", alpha=0.18, label="selected")
        if comparison_x_plot is not None:
            hist_ax.hist(comparison_x_plot, bins="auto", color="0.35", alpha=0.16, label=comparison_label)
        hist_ax.set_ylabel("Count", color="0.45")
        hist_ax.tick_params(axis="y", colors="0.45")
        hist_ax.legend(loc="upper right")

        for ax, state_idx in zip(axes[:-1], component_indices):
            if comparison_x_plot is not None:
                ax.scatter(
                    comparison_x_plot,
                    comparison_x0_plot[:, state_idx],
                    facecolors="none",
                    edgecolors="0.35",
                    s=28,
                    alpha=0.65,
                    linewidths=0.8,
                    label=comparison_label,
                )
            ax.scatter(
                x_plot,
                x0_plot[:, state_idx],
                c=colors[state_idx],
                s=32,
                alpha=0.85,
                edgecolors="none",
                label="selected",
            )
            ax.plot(x_plot, x0_plot[:, state_idx], color=colors[state_idx], alpha=0.25, linewidth=1.0)
            ax.plot(x_plot, np.full_like(x_plot, ax.get_ylim()[0]), "|", color="0.25", alpha=0.55)
            ax.set_ylabel(state_labels[state_idx])
            ax.grid(True, alpha=0.3)

        if comparison_x_plot is not None:
            axes[-1].scatter(
                comparison_x_plot[comparison_valid],
                comparison_u0_plot[comparison_valid],
                facecolors="none",
                edgecolors="0.35",
                s=34,
                alpha=0.7,
                linewidths=0.8,
                label=f"{comparison_label} first MPC control",
            )
        axes[-1].scatter(
            x_plot[valid],
            u0_plot[valid],
            c="tab:purple",
            s=36,
            alpha=0.9,
            edgecolors="none",
            label="selected first MPC control",
        )
        if np.any(~valid):
            axes[-1].scatter(
                x_plot[~valid],
                np.zeros(np.sum(~valid)),
                c="tab:red",
                s=40,
                marker="x",
                label="failed MPC solve",
            )
        if comparison_x_plot is not None and np.any(~comparison_valid):
            axes[-1].scatter(
                comparison_x_plot[~comparison_valid],
                np.zeros(np.sum(~comparison_valid)),
                c="0.35",
                s=40,
                marker="x",
                label=f"failed {comparison_label} MPC solve",
            )
        axes[-1].axhline(0.0, color="k", linestyle="--", alpha=0.3)
        axes[-1].plot(x_plot, np.full_like(x_plot, axes[-1].get_ylim()[0]), "|", color="0.25", alpha=0.55)
        if comparison_x_plot is not None:
            axes[-1].plot(
                comparison_x_plot,
                np.full_like(comparison_x_plot, axes[-1].get_ylim()[0]),
                "|",
                color="0.35",
                alpha=0.35,
            )
        axes[-1].set_ylabel("First Control (N)")
        axes[-1].set_xlabel(f"Initial {state_labels[x_axis_idx]}")
        axes[-1].grid(True, alpha=0.3)
        axes[-1].legend(loc="best")

        fig.suptitle("Initial State Distribution and First MPC Control")
        fig.tight_layout()
        if show:
            plt.show()

        return fig, axes
    
    def bilevel_active_set(
        self,
        x_max,
        slack_penalty="quadratic",
        return_solver=False,
        complementarity_method="ccopt",
        scholtes_relaxations=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    ):
        """
        Bilevel active-set method for finding a feasible initial state for the MPC problem.

        ``complementarity_method`` selects either the native ``"ccopt"``
        complementarity solver or a ``"scholtes"`` relaxation homotopy solved
        with IPOPT.  ``scholtes_relaxations`` must be a strictly decreasing
        sequence of positive relaxation parameters. Solver bundles requested
        with ``return_solver=True`` are cached on this model instance.
        """
        slack_penalty = str(slack_penalty).lower()
        if slack_penalty not in {"l1", "quadratic"}:
            raise ValueError("slack_penalty must be either 'l1' or 'quadratic'.")
        complementarity_method = str(complementarity_method).lower().replace("-", "_")
        if complementarity_method == "scholtes_ipopt":
            complementarity_method = "scholtes"
        if complementarity_method not in {"ccopt", "scholtes"}:
            raise ValueError(
                "complementarity_method must be either 'ccopt' or 'scholtes'."
            )

        # Solver construction is expensive and its symbolic structure depends
        # only on these settings. Keep the cache local to this model instance,
        # so solvers are never shared across horizons or CartPole objects.
        effective_x_max = (
            np.array(
                [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound],
                dtype=float,
            )
            if x_max is None
            else np.asarray(x_max, dtype=float)
        )
        cache_key = (
            slack_penalty,
            complementarity_method,
            tuple(float(value) for value in scholtes_relaxations),
            tuple(effective_x_max.reshape(-1)),
        )
        solver_cache = getattr(self, "_bilevel_solver_cache", None)
        if solver_cache is None:
            solver_cache = {}
            self._bilevel_solver_cache = solver_cache
        if return_solver and cache_key in solver_cache:
            self._last_bilevel_solver_cache_hit = True
            return solver_cache[cache_key]
        self._last_bilevel_solver_cache_hit = False

        def weighted_slack_cost(slacks, weights):
            regularization = 1e-6 * ca.dot(slacks, slacks)
            if slack_penalty == "l1":
                return ca.dot(weights, slacks)
            return 0.5 * slacks.T @ ca.diag(weights) @ slacks

        x = ca.SX.sym('x', 4)
        u = ca.SX.sym('u', 1)
        x0 = ca.SX.sym('x0', 4) # initial state parameter
        xr = ca.DM([0.0, 0.0, 0.0, 0.0])
        # A, B = self.lin_dyn(xr, ca.DM([0]))
        # E = solve_continuous_are(A.full(), B.full(), self.Q.full(), self.R.full())

        x_next = self.mpc_step(x, u)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
        F = ca.Function('F', [x, u], [x_next, l])

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

        if x_max is None:
            x_max = np.array([self.p_bound, self.v_bound, self.theta_bound, self.omega_bound], dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (4,):
                raise ValueError("x_max must have shape (4,).")

        x_min = -x_max
        u_max = np.array([self.u_bound], dtype=float)
        u_min = -u_max

        Xk = ca.SX.sym('X0', 4)
        w += [Xk]
        lbw += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
        ubw += [ca.inf, ca.inf, ca.inf, ca.inf]
        w0 += [0.0, 0.0, np.pi, 0.0]
        w_mpc += [Xk]
        
        s_x0 = ca.SX.sym('slack_x0', 4, 2)
        s_x0_var = ca.vec(s_x0)
        s_x0_start = len(lbw)
        w += [s_x0_var]
        lbw += [0.0 for _ in range(8)]
        ubw += [ca.inf for _ in range(8)]
        w0 += [0.0 for _ in range(8)]
        slack_indices.extend(range(s_x0_start, s_x0_start + s_x0_var.numel()))
        
        g += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
        lbg += [0.0 for _ in range(8)]
        ubg += [0.0 for _ in range(8)]

        slacks = [s_x0_var]
        # Each residual block stores upper residuals first, then lower residuals.
        # Keep the block sizes so random spiking can choose at most one side per bound.
        slack_block_sizes = [4]

        for k in range(self.N):
            uk = ca.SX.sym('u_' + str(k), 1)
            w += [uk]
            lbw += [-ca.inf]
            ubw += [ca.inf]
            w0 += [0.0]
            w_mpc += [uk]

            s_uk = ca.SX.sym('slack_u_' + str(k), 1, 2)
            s_uk_var = ca.vec(s_uk)
            s_uk_start = len(lbw)
            w += [s_uk_var]
            lbw += [0.0, 0.0]
            ubw += [ca.inf, ca.inf]
            w0 += [0.0, 0.0]
            slack_indices.extend(range(s_uk_start, s_uk_start + s_uk_var.numel()))

            g += [uk - u_max + s_uk[:, 0], -uk + u_min + s_uk[:, 1]]
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]
            
            Xk_next, _ = F(Xk, uk)

            Xk = ca.SX.sym('X_' + str(k + 1), 4)
            w += [Xk]
            lbw += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
            ubw += [ca.inf, ca.inf, ca.inf, ca.inf]
            w0 += [0.0, 0.0, np.pi, 0.0]
            w_mpc += [Xk]
            
            g += [Xk_next - Xk]  # dynamics constraint
            lbg += [0.0 for _ in range(4)]
            ubg += [0.0 for _ in range(4)]

            s_xk = ca.SX.sym('slack_x_' + str(k), 4, 2)
            s_xk_var = ca.vec(s_xk)
            s_xk_start = len(lbw)
            w += [s_xk_var]
            lbw += [0.0 for _ in range(8)]
            ubw += [ca.inf for _ in range(8)]
            w0 += [0.0 for _ in range(8)]
            slack_indices.extend(range(s_xk_start, s_xk_start + s_xk_var.numel()))

            g += [Xk - x_max + s_xk[:, 0], -Xk + x_min + s_xk[:, 1]]
            lbg += [0.0 for _ in range(8)]
            ubg += [0.0 for _ in range(8)]

            slacks.append(s_uk_var)
            slacks.append(s_xk_var)
            # Input has one bounded scalar; state has four bounded scalars.
            slack_block_sizes.append(1)
            slack_block_sizes.append(4)

        # Add the slack penalty to the objective function
        rho, _ = self.spiky_residual_weights(
                slack_block_sizes,
                min_spikes=1,
                max_spikes=1,
                base_scale=1e-6,
                spike_factor_range=(1e0, 1e2),
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
        
        # Parameters exposed to callers.  The Scholtes relaxation parameter is
        # appended privately by ScholtesHomotopySolver below.
        if return_solver:
            user_params = ca.vertcat(x0, rho_param)
        else:
            user_params = x0
            
        varvar = 0

        if len(slack_indices) != lam_g.numel():
            raise RuntimeError("One residual is required per inequality multiplier.")

        # Create the NLP solver
        print(f"Creating {complementarity_method} bilevel NLP solver...")
        if complementarity_method == "ccopt":
            ind_cc = [
                [lam_g_start + i, slack_indices[i]]
                for i in range(lam_g.numel())
            ]
            cctypes = [varvar] * lam_g.numel()
            nlp = {
                'f': J,
                'x': ca.vertcat(*w),
                'g': ca.vertcat(*g),
                'p': user_params,
            }
            opts = {
                "expand": False,
                "print_time": False,
                # ccopt delegates NLP solves to MadNLP; bound_relax_factor=0.0 is
                # important for complementarity formulations.
                "cc_pairs": ind_cc,
                "cc_types": cctypes,
                "madnlp.print_level": 6,
                "madnlp.max_iter": 2000,
                "madnlp.tol": 1e-6,
                "madnlp.bound_relax_factor": 0.0,
                "madnlp.linear_solver": "Ma97Solver",
                "ccopt.relaxation_update.TYPE": "RolloffRelaxationUpdate",
                "ccopt.relaxation_update.rolloff_point": 1e-3,
                "ccopt.relaxation_update.rolloff_slope": 1.5,
                "ccopt.relaxation_update.sigma_min": 1e-6,
            }
            bilevel_solver = ca.nlpsol("bilevel_solver", "ccopt", nlp, opts)
        else:
            relaxation = ca.SX.sym("scholtes_relaxation")
            complementarity_products = lam_g * ca.vertcat(*slacks)
            g += [complementarity_products - relaxation]
            lbg += [-ca.inf] * lam_g.numel()
            ubg += [0.0] * lam_g.numel()
            nlp = {
                'f': J,
                'x': ca.vertcat(*w),
                'g': ca.vertcat(*g),
                'p': ca.vertcat(user_params, relaxation),
            }
            opts = {
                "expand": False,
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 5000,
                "ipopt.tol": 1e-8,
                "ipopt.acceptable_tol": 1e-6,
            }
            ipopt_solver = ca.nlpsol(
                "bilevel_scholtes_solver", "ipopt", nlp, opts
            )
            bilevel_solver = ScholtesHomotopySolver(
                ipopt_solver, user_params.numel(), scholtes_relaxations
            )
        
        mpc_vars = ca.Function('mpc_vars', [ca.vertcat(*w)], [ca.vertcat(*w_mpc)])
        lambda_vars = ca.Function('lambda_vars', [ca.vertcat(*w)], [lam_h, lam_g])
        slack_vars = ca.Function('slack_vars', [ca.vertcat(*w)], [ca.vertcat(*slacks)])
        
        if return_solver:
            solver_bundle = (
                bilevel_solver,
                w0,
                lbw,
                ubw,
                lbg,
                ubg,
                slack_block_sizes,
                mpc_vars,
                lambda_vars,
                slack_vars,
            )
            solver_cache[cache_key] = solver_bundle
            return solver_bundle
        
        # Solve the bilevel problem
        print("Solving bilevel problem...")
        # Sample an initial state
        x0_num = ca.DM([0.0, 0.0, np.pi, 0.0])
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
        
        x0_feasible = sol['x'][0:4].full().flatten()
        print(f"Feasible initial state found: {x0_feasible}")
        
        # From the optimal solution, extract the mpc solution
        mpc_solution = mpc_vars(sol['x']).full().flatten()
        
        return x0_feasible, mpc_solution

    def seek_x0(
        self,
        N,
        x_max=None,
        close_to_bounds=True,
        u0_active_bound=False,
        plot_results=False,
        return_u_seq=False,
        slack_penalty="quadratic",
        print_stationarity_perturbation=False,
        return_traj=False,
    ):
        """Find a feasible initial state with the paper's slacked auxiliary NLP.

        The returned trajectory is feasible for the original MPC constraints.
        It is generally stationary for the perturbed objective characterized in
        the paper, and is stationary for the original objective only when the
        reported perturbation vanishes.

        Parameters
        ----------
        N : int
            MPC horizon length.
        x_max : array_like, optional
            Maximum absolute values for the state components. If None, defaults to
            ``[self.p_bound, self.v_bound, self.theta_bound, self.omega_bound]``.
        close_to_bounds : bool, optional
            If True, both state and input inequalities may receive positive
            sampled activity weights. If False, only input inequalities are
            targeted. All residual equations remain exact in either case.
        u0_active_bound : bool, optional
            If True, one side of the first-input bound is always included in
            the sampled target set (it is penalized, not imposed as an equality).
        plot_results : bool, optional
            If True, plots the initial state and first control input.
        return_u_seq : bool, optional
            If True, returns the full control sequence from the MPC solution.
        return_traj : bool, optional
            If True, returns the full state and control trajectories from the MPC solution.
        slack_penalty : {"quadratic"}, optional
            Kept as an explicit argument for API compatibility. The paper's
            objective is exactly ``0.5 * s.T @ Q @ s``; L1 is rejected.
        print_stationarity_perturbation : bool, optional
            If True, display c* and whether its relative residual is within the
            numerical zero tolerance.
        """
        slack_penalty = str(slack_penalty).lower()
        if slack_penalty != "quadratic":
            raise ValueError(
                "The paper-compliant slacked formulation only supports "
                "slack_penalty='quadratic'."
            )

        def weighted_slack_cost(slacks, weights):
            # Exactly 1/2 ||s||_Q^2, so its slack gradient is Qs as in the
            # derivation in the paper (with no unreported regularizer).
            return 0.5 * ca.dot(ca.DM(weights), slacks**2)

        if N < 1:
            raise ValueError("N must be at least 1.")

        x = ca.SX.sym('x', 4)
        u = ca.SX.sym('u', 1)
        xr = ca.DM([0.0, 0.0, 0.0, 0.0])
        A, B = self.lin_dyn(xr, ca.DM([0]))
        E = solve_continuous_are(A.full(), B.full(), self.Q.full(), self.R.full())

        x_next = self.mpc_step(x, u)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
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

        state_bounds = np.array(
            [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound],
            dtype=float,
        )
        if x_max is None:
            x_max = state_bounds.copy()
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (4,):
                raise ValueError("x_max must have shape (4,).")
            if np.any(~np.isfinite(x_max)) or np.any(x_max <= 0.0):
                raise ValueError("x_max must contain positive finite values.")
            if np.any(x_max > state_bounds + 1e-12):
                raise ValueError("x_max cannot exceed the original MPC state bounds.")

        state_min = -state_bounds
        u_max = np.array([self.u_bound], dtype=float)
        u_min = -u_max

        Xk = ca.SX.sym('X0', 4)
        w += [Xk]
        lbw += [-x_max[0], -x_max[1], -x_max[2], -x_max[3]]
        ubw += [x_max[0], x_max[1], x_max[2], x_max[3]]
        w0 += [0.0, 0.0, np.pi, 0.0]
        w_mpc += [Xk]
        
        # x0 is free in the auxiliary problem, but its original state-bound
        # inequalities remain in g(w) and therefore require slacks too.
        s_x0 = ca.SX.sym('slack_x0', 4, 2)
        s_x0_var = ca.vec(s_x0)
        w += [s_x0_var]
        lbw += [0.0 for _ in range(8)]
        ubw += [ca.inf for _ in range(8)]
        w0 += [0.0 for _ in range(8)]
        constraints += [
            Xk - state_bounds + s_x0[:, 0],
            -Xk + state_min + s_x0[:, 1],
        ]
        lbg += [0.0 for _ in range(8)]
        ubg += [0.0 for _ in range(8)]

        X_syms = [Xk]
        U_syms = []
        x_slacks = [s_x0_var]
        u_slacks = []
        # Each residual block stores upper residuals first, then lower residuals.
        # Keep the block sizes so random spiking can choose at most one side per bound.
        x_slack_block_sizes = [4]
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

            Xk = ca.SX.sym('X_' + str(k + 1), 4)
            w += [Xk]
            lbw += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
            ubw += [ca.inf, ca.inf, ca.inf, ca.inf]
            w0 += [0.0, 0.0, 0.0, 0.0]
            w_mpc += [Xk]

            s_xk = ca.SX.sym('slack_x_' + str(k), 4, 2)
            s_xk_var = ca.vec(s_xk)
            w += [s_xk_var]
            lbw += [0.0 for _ in range(8)]
            ubw += [ca.inf for _ in range(8)]
            w0 += [0.0 for _ in range(8)]

            constraints += [
                Xk - state_bounds + s_xk[:, 0],
                -Xk + state_min + s_xk[:, 1],
            ]
            # The paper requires g(w) + s = 0 for every inequality residual.
            lbg += [0.0 for _ in range(8)]
            ubg += [0.0 for _ in range(8)]

            constraints += [Xk_next - Xk]
            lbg += [0.0 for _ in range(4)]
            ubg += [0.0 for _ in range(4)]

            J += l_k
            X_syms.append(Xk)
            U_syms.append(uk)
            x_slacks.append(s_xk_var)
            u_slacks.append(s_uk_var)
            # Input has one bounded scalar; state has four bounded scalars.
            x_slack_block_sizes.append(4)
            u_slack_block_sizes.append(1)

        terminal_value = (Xk - xr).T @ ca.DM(E) @ (Xk - xr)
        alpha = 1e-1
        J += terminal_value
        # constraints += [terminal_value]
        # lbg += [-ca.inf]
        # ubg += [alpha]

        x_slacks_vec = ca.vertcat(*x_slacks)
        u_slacks_vec = ca.vertcat(*u_slacks)

        # Q has positive entries only for the sampled target set T.  Sampling
        # at most n_w - m_h targets enforces m_h + |T| <= n_w.
        block_sizes = x_slack_block_sizes + u_slack_block_sizes
        residual_pairs = []
        offset = 0
        for block_size in block_sizes:
            for component in range(block_size):
                residual_pairs.append(
                    (offset + component, offset + block_size + component)
                )
            offset += 2 * block_size

        n_state_pairs = sum(x_slack_block_sizes)
        eligible_pairs = np.arange(len(residual_pairs), dtype=int)
        if not close_to_bounds:
            # Retain the historical option's intent: target only input bounds.
            eligible_pairs = eligible_pairs[n_state_pairs:]

        n_w = 4 * (N + 1) + N
        m_h = 4 * N
        max_targets = min(n_w - m_h, eligible_pairs.size)
        if max_targets < 1:
            raise RuntimeError("No inequality constraints are available for targeting.")

        forced_pair = n_state_pairs if u0_active_bound else None
        if forced_pair is not None and forced_pair not in eligible_pairs:
            eligible_pairs = np.append(eligible_pairs, forced_pair)
            max_targets = min(n_w - m_h, eligible_pairs.size)

        num_targets = np.random.randint(1, max_targets + 1)
        if forced_pair is None:
            target_pairs = np.random.choice(
                eligible_pairs, size=num_targets, replace=False
            )
        else:
            remaining = eligible_pairs[eligible_pairs != forced_pair]
            extra_count = num_targets - 1
            extras = np.random.choice(remaining, size=extra_count, replace=False)
            target_pairs = np.r_[forced_pair, extras]

        rho = np.zeros(2 * sum(block_sizes), dtype=float)
        target_residuals = np.array(
            [residual_pairs[pair][np.random.randint(2)] for pair in target_pairs],
            dtype=int,
        )
        rho[target_residuals] = 10.0 ** np.random.uniform(
            0.0, 2.0, size=target_residuals.size
        )
        rho_x = rho[:x_slacks_vec.numel()]
        rho_u = rho[x_slacks_vec.numel():]
        J += weighted_slack_cost(x_slacks_vec, rho_x)
        J += weighted_slack_cost(u_slacks_vec, rho_u)
        self.last_slacked_problem_stats = {
            "n_w": int(n_w),
            "m_h": int(m_h),
            "target_count": int(target_residuals.size),
            "target_indices": target_residuals.copy(),
        }
        
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
                # "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
                # "linear_solver": "ma27",
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

        # Evaluate c* for every solve so batch statistics do not depend on
        # whether verbose perturbation printing was requested.
        slack_values = np.asarray(opt_slacks.full(), dtype=float).reshape(-1)
        inactive_slack_values = np.where(
            np.abs(slack_values) > 1e-8, slack_values, 0.0
        )
        constraint_jacobian_at_opt = slacked_constraint_jacobian(
            sol['x']
        ).full()
        constraint_rows = []
        for dec_idx in slack_decision_indices:
            rows = np.flatnonzero(
                np.abs(constraint_jacobian_at_opt[:, dec_idx]) > 1e-12
            )
            if rows.size != 1:
                raise RuntimeError(
                    f"Slack decision variable {dec_idx} appears in constraint rows {rows}"
                )
            constraint_rows.append(rows[0])
        constraint_rows = np.asarray(constraint_rows, dtype=int)
        slack_and_x0_indices = np.concatenate(
            [np.arange(4), slack_decision_indices]
        )
        y_decision_indices = np.setdiff1d(
            all_decision_indices, slack_and_x0_indices
        )
        g_jacobian_y = constraint_jacobian_at_opt[
            np.ix_(constraint_rows, y_decision_indices)
        ].T
        weighted_inactive_slacks = rho * inactive_slack_values
        c_star = -(g_jacobian_y @ weighted_inactive_slacks)
        c_star_norm = float(np.linalg.norm(c_star))
        c_star_scale = (
            np.linalg.norm(g_jacobian_y, ord=np.inf)
            * np.linalg.norm(weighted_inactive_slacks, ord=np.inf)
        )
        c_star_relative_residual = c_star_norm / max(1.0, c_star_scale)
        self.last_slacked_c_star = np.asarray(c_star, dtype=float).reshape(-1)
        self.last_slacked_c_star_norm = c_star_norm
        self.last_slacked_c_star_is_zero = bool(
            c_star_relative_residual <= 1e-6
        )
        
        if print_stationarity_perturbation:
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
            
            # Determine the y variables. They are all variables exept x0 and the slacks
            slack_and_x0_indices = np.concatenate([np.arange(4), slack_decision_indices])
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

            print("c*:", self.last_slacked_c_star)
            print("||c*||:", self.last_slacked_c_star_norm)
            print("c* numerically zero:", self.last_slacked_c_star_is_zero)
            
            # Check if the g_jacobian_y_I matrix is full column rank
            # if np.linalg.matrix_rank(g_jacobian_y_I) < g_jacobian_y_I.shape[1]:
            #     print("g_jacobian_y_I is not full column rank.")

            # if np.linalg.matrix_rank(g_jacobian_y_I @ Q_I) < g_jacobian_y_I.shape[1]:
            #     print("g_jacobian_y_I @ Q_I is not full column rank.")

        if plot_results:
            x_pred = np.array(X_opt.full())
            u_pred = np.array(U_opt.full()).flatten()

            tx = np.arange(x_pred.shape[1]) * self.dt
            tu = np.arange(u_pred.shape[0]) * self.dt

            fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=False)
            axes[0].plot(tx, x_pred[0], "b-o", linewidth=1.5, markersize=3)
            axes[0].set_ylabel("Position (m)")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(tx, x_pred[1], "r-o", linewidth=1.5, markersize=3)
            axes[1].set_ylabel("Velocity (m/s)")
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(tx, x_pred[2], "g-o", linewidth=1.5, markersize=3)
            axes[2].axhline(0.0, color="k", linestyle="--", alpha=0.4)
            axes[2].set_ylabel("Angle (rad)")
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(tx, x_pred[3], "c-o", linewidth=1.5, markersize=3)
            axes[3].set_ylabel("Angular vel (rad/s)")
            axes[3].grid(True, alpha=0.3)

            axes[4].plot(tu, u_pred, "k-", linewidth=1.8)
            axes[4].set_ylim(-self.u_bound * 1.2, self.u_bound * 1.2)
            axes[4].set_ylabel("Control")
            axes[4].set_xlabel("Time (s)")
            axes[4].grid(True, alpha=0.3)

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

    def spiky_residual_weights(
        self,
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

    def spiky_array(
        self,
        n,
        min_spikes=1,
        max_spikes=3,
        base_scale=1.0,
        spike_factor_range=(50, 1000),
        positive=True,
        spike_probability=0.4,
    ):
        """Generate an array that may contain no spikes or a random number of spikes."""
        x = np.random.exponential(base_scale, size=n) if positive else np.random.normal(0, base_scale, size=n)

        if n <= 0 or np.random.random() > spike_probability:
            return x, np.array([], dtype=int)

        min_spikes = max(1, min_spikes)
        max_spikes = min(max_spikes, n)
        if max_spikes < min_spikes:
            return x, np.array([], dtype=int)

        k = np.random.randint(min_spikes, max_spikes + 1)
        spike_idx = np.random.choice(n, size=k, replace=False)
        factors = np.random.uniform(*spike_factor_range, size=k)
        x[spike_idx] *= factors

        return x, spike_idx

    def _linearized_ocp_at_solution(self, solution):
        """Build a local affine OCP around a feasible nonlinear trajectory.

        Linearizing at every nominal ``(x[k], u[k])`` pair gives an LTV affine
        model that reproduces the nominal nonlinear trajectory exactly. Thus,
        the seed remains feasible for its local linearized OCP up to numerical
        solver tolerances.
        """
        X_opt, U_opt = self.extract_traj(solution['x'])
        X_opt = X_opt.full()
        U_opt = U_opt.full()
        A_seq = []
        B_seq = []
        for k in range(self.N):
            # dx[k+1] ~= A[k] dx[k] + B[k] du[k]
            A, B = self.discrete_lin_dyn(X_opt[:, k], U_opt[:, k])
            A_seq.append(A.full())
            B_seq.append(B.full())
            
        # After collecting the linearized dynamics, compute the condensed matrices for the affine MPC representation.
        # Omega = [A[0], A[1] @ A[0], A[2] @ A[1] @ A[0], ..., A[N-1] @ ... @ A[0]]
        # Gamma = [B[0], 0, 0, ..., 0;
        #          A[1] @ B[0], B[1], 0, ..., 0;
        #          A[2] @ A[1] @ B[0], A[2] @ B[1], B[2], ..., 0;
        #         ...
        #          A[N-1] @ ... @ A[1] @ B[0], A[N-1] @ ... @ A[2] @ B[1], ..., B[N-1]]
        state_dim = 4
        input_dim = 1
        Omega = np.zeros((self.N * state_dim, state_dim))
        Gamma = np.zeros((self.N * state_dim, self.N * input_dim))
        for k in range(self.N):
            # dx[k+1] = A[k] ... A[0] dx[0] +
            #             sum_j A[k] ... A[j+1] B[j] du[j].
            # Building the products recursively handles k=0 without calling
            # np.linalg.multi_dot on a single matrix and preserves the
            # required reverse-time multiplication order.
            state_transition = np.eye(state_dim)
            for transition_index in range(k + 1):
                state_transition = A_seq[transition_index] @ state_transition
            row_slice = slice(k * state_dim, (k + 1) * state_dim)
            Omega[row_slice, :] = state_transition

            for j in range(k + 1):
                input_transition = np.eye(state_dim)
                for transition_index in range(j + 1, k + 1):
                    input_transition = (
                        A_seq[transition_index] @ input_transition
                    )
                column_slice = slice(j * input_dim, (j + 1) * input_dim)
                Gamma[row_slice, column_slice] = (
                    input_transition @ B_seq[j]
                )
        
        # The constraints are already linear from the original problem.
        # Collect the state bounds in the form Hx x <= h_x and Hu u <= h_u for the linearized OCP.
        Hx = np.vstack([np.eye(4), -np.eye(4)])
        h_x = np.hstack([self.x_max, -self.x_min])
        Hu = np.vstack([np.eye(1), -np.eye(1)])
        h_u = np.hstack([self.u_bound, self.u_bound])
        
        # We can condense the constraints into the form F = [Hx @ Omega; 0] and G = [Hx @ Gamma; Hu] for the linearized OCP.
        # We also include the known term b = [h_x; h_u] for the linearized OCP.
        # Create a block diagonal matrix for the state constraints over the horizon.
        Hx_block = np.kron(np.eye(self.N), Hx)
        Hu_block = np.kron(np.eye(self.N), Hu)
        h_x_block = np.tile(h_x, self.N)
        h_u_block = np.tile(h_u, self.N)
        
        G = np.vstack([Hx_block @ Gamma, Hu_block])
        F = np.vstack([Hx_block @ Omega, np.zeros((Hu_block.shape[0], Omega.shape[1]))])
        # Omega and Gamma propagate perturbations around the nominal solution,
        # so the right-hand side must contain the remaining constraint margin.
        nominal_states = X_opt[:, 1:].T.reshape(-1)
        nominal_inputs = U_opt.T.reshape(-1)
        b = np.hstack([
            h_x_block - Hx_block @ nominal_states,
            h_u_block - Hu_block @ nominal_inputs,
        ])
        min_nominal_margin = float(np.min(b))
        if min_nominal_margin < -BOUNDARY_LP_FEASIBILITY_TOL:
            raise RuntimeError(
                "The nominal trajectory is not feasible for its linearized "
                f"OCP: minimum state/input margin is {min_nominal_margin:.3e}."
            )
        # Preserve the construction guarantee at active bounds when the NLP
        # solution is only outside by solver-scale numerical noise.
        b = np.maximum(b, 0.0)

        return {
            "G": G,
            "F": F,
            "b": b,
            "X_nominal": X_opt,
            "U_nominal": U_opt,
        }

    def _linearized_boundary_step(
        self,
        dx0,
        direction,
        linear_ocp,
        initial_state_bounds,
    ):
        """Find the linearized feasible-set boundary along one direction.

        For the line ``x(alpha) = x + alpha * direction``, solve

        ``maximize alpha``
        ``subject to the condensed affine MPC constraints and alpha >= 0``.

        CasADi's conic interface minimizes, so the objective coefficient of
        ``alpha`` is set to ``-1``. The returned value is the largest feasible
        nonnegative step in the supplied direction.
        """
        G = np.asarray(linear_ocp["G"], dtype=float)
        F = np.asarray(linear_ocp["F"], dtype=float)
        b = np.asarray(linear_ocp["b"], dtype=float).reshape(-1)

        dx0 = np.asarray(dx0, dtype=float).reshape(-1)
        direction = np.asarray(direction, dtype=float).reshape(-1)
        initial_state_bounds = np.asarray(
            initial_state_bounds,
            dtype=float,
        ).reshape(-1)

        assert dx0.size == 4
        assert direction.size == 4
        assert initial_state_bounds.size == 4

        seed = np.asarray(
            linear_ocp["X_nominal"],
            dtype=float,
        )[:, 0].reshape(-1)

        Fd = F @ direction
        rhs_ocp = b - F @ dx0 + BOUNDARY_LP_FEASIBILITY_TOL

        assert rhs_ocp.shape == (G.shape[0],)
        assert Fd.shape == (G.shape[0],)

        A_ocp = np.column_stack([
            G,
            Fd,
        ])

        zero_controls = np.zeros((4, G.shape[1]))

        A_box = np.vstack([
            np.column_stack([
                zero_controls,
                direction,
            ]),
            np.column_stack([
                zero_controls,
                -direction,
            ]),
        ])

        uba_box = np.concatenate([
            initial_state_bounds - seed - dx0,
            initial_state_bounds + seed + dx0,
        ]) + BOUNDARY_LP_FEASIBILITY_TOL

        A = np.vstack([
            A_ocp,
            A_box,
        ])

        uba = np.concatenate([
            rhs_ocp,
            uba_box,
        ])

        lba = np.full(A.shape[0], -np.inf)
        # The decision vector is [du, alpha], so the number of variables is n_du + 1.
        n_variables = A.shape[1]
        n_constraints = A.shape[0]

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
                "a": ca.Sparsity.dense(n_constraints, n_variables),
            }
            options = {
                "print_time": False,
                "error_on_fail": False,
                # Solve the LP through IPOPT rather than qpOASES: the "nlpsol"
                # conic plugin casts the QP/LP as a (trivial) NLP and hands it
                # to the named nlpsol solver.
                # "nlpsol": "ipopt",
                # "nlpsol_options": {
                #     "print_time": False,
                #     "ipopt.print_level": 0,
                #     "ipopt.hessian_constant": "yes",
                #     "ipopt.least_square_init_primal": "yes",
                #     "ipopt.least_square_init_duals": "yes",
                #     "ipopt.mehrotra_algorithm": "yes",
                #     "ipopt.grad_f_constant": "yes",
                #     "ipopt.jac_c_constant": "yes",
                #     "ipopt.jac_d_constant": "yes",
                #     "ipopt.max_iter": 100,
                # },
                # "nlpsol": "qpoases",
                # "nlpsol_options": {
                #     "print_time": False,
                #     "qpoases.printLevel": "none",
                # },
                "highs": {
                    "output_flag": False,  # HiGHS output
                    "log_to_console": False,
                }
            }
            solver = ca.conic(
                (
                    f"linearized_boundary_{id(self)}_"
                    f"{n_constraints}_{n_variables}"
                ),
                "highs",
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

        stats = solver.stats()

        if not stats.get("success", False):
            print("HiGHS failure")
            print("return_status:", stats.get("return_status"))
            print("unified_return_status:", stats.get("unified_return_status"))
            return None

        solution = result["x"].full().reshape(-1)
        alpha = float(solution[-1])
        if not np.isfinite(alpha):
            return None
        # IPOPT can violate the alpha >= 0 bound by a small feasibility
        # tolerance. Clamp only that numerical residue; a material violation
        # means this direction is not a valid Hit-and-Run chord.
        if alpha < -1e-8:
            return None
        return max(alpha, 0.0)

    def _build_feasible_projection_problem(self):
        """Build the weighted, cost-aware nonlinear projection problem.

        With ``w = [x0, u0, x1, ..., uN-1, xN]``, target ``x0_bar``,
        and state weight ``Qx``, this is

        ``min f(w) + 0.5*(x0 - x0_bar)' Qx (x0 - x0_bar)``

        subject to the nonlinear dynamics ``h_tilde(w) = 0`` and every
        original state/input inequality ``g(w) <= 0``.
        """
        sym = getattr(ca, self.sym_type)
        target = sym.sym('projection_target', 4)
        Qx_flat = sym.sym('projection_Qx', 16)
        Qx = ca.reshape(Qx_flat, 4, 4)
        X = sym.sym('projection_X', 4, self.N + 1)
        U = sym.sym('projection_U', 1, self.N)
        # Keep exactly the same [x0, u0, x1, ..., uN-1, xN] ordering used by
        # the MPC solver, allowing a feasible MPC solution to be a warm start.
        decision = ca.veccat(X[:, 0], ca.vertcat(U, X[:, 1:]))
        dynamics_constraints = []
        for k in range(self.N):
            dynamics_constraints.append(
                self.mpc_step(X[:, k], U[:, k]) - X[:, k + 1]
            )
        h_tilde = ca.vertcat(*dynamics_constraints)
        inequalities = self.g(decision)
        constraints = ca.vertcat(h_tilde, inequalities)

        displacement = X[:, 0] - target
        objective = (
            self.f(decision)
            + 0.5 * ca.mtimes([displacement.T, Qx, displacement])
        )
        parameters = ca.vertcat(target, Qx_flat)

        problem = {
            'f': objective,
            'x': decision,
            'g': constraints,
            'p': parameters,
        }
        opts = {
            "print_time": False,
            "expand": True,
            "ipopt": {"print_level": 0, 
                      "max_iter": 1000, 
                      "tol": 1e-6,
                      "mu_strategy": "adaptive",},
        }
        solver = ca.nlpsol("feasible_projection_solver", "ipopt", problem, opts)
        n_equalities = int(h_tilde.numel())
        n_inequalities = int(inequalities.numel())
        n_variables = int(decision.numel())
        return {
            "solver": solver,
            "lbx": np.full(n_variables, -np.inf),
            "ubx": np.full(n_variables, np.inf),
            "lbg": np.r_[
                np.zeros(n_equalities),
                np.full(n_inequalities, -np.inf),
            ],
            "ubg": np.zeros(n_equalities + n_inequalities),
            "constraints": ca.Function(
                "projection_constraints",
                [decision],
                [constraints],
            ),
            "n_equalities": n_equalities,
        }

    def project_onto_feasible_set(
        self,
        x_target,
        Qx=None,
        feasible_reference=None,
        reference_solution=None,
        return_solution=False,
    ):
        """Solve the weighted, MPC-cost-aware nonlinear projection problem."""
        x_target = np.asarray(x_target, dtype=float).reshape(4)
        Qx = self._validate_projection_weight(Qx)
        
        if reference_solution is None:
            # Obtain a complete feasible trajectory, not only a feasible x0,
            # to initialize every state and input in the projection NLP.
            reference = np.zeros(4) if feasible_reference is None else np.asarray(
                feasible_reference, dtype=float
            ).reshape(4)
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
                raise RuntimeError("The projection warm-start state is not MPC feasible.") from exc
            if not self.solver.stats().get("success", False):
                raise RuntimeError("The projection warm-start state is not MPC feasible.")

        if self._feasible_projection_problem is None:
            # The NLP structure depends only on the model and horizon.
            self._feasible_projection_problem = self._build_feasible_projection_problem()
        projection = self._feasible_projection_problem
        parameters = np.r_[x_target, Qx.reshape(-1, order="F")]
        sol = projection["solver"](
            x0=reference_solution['x'],
            lbx=projection["lbx"],
            ubx=projection["ubx"],
            lbg=projection["lbg"],
            ubg=projection["ubg"],
            p=parameters,
        )
        if not projection["solver"].stats().get("success", False):
            status = projection["solver"].stats().get("return_status", "unknown")
            raise RuntimeError(f"Nonlinear feasible-set projection failed ({status}).")

        residuals = projection["constraints"](sol["x"]).full().reshape(-1)
        n_equalities = projection["n_equalities"]
        equality_residual = residuals[:n_equalities]
        inequality_residual = residuals[n_equalities:]
        if (
            np.linalg.norm(equality_residual, ord=np.inf) > 1e-6
            or np.max(inequality_residual, initial=-np.inf) > 1e-6
        ):
            raise RuntimeError(
                "Nonlinear feasible-set projection returned a "
                "constraint-violating trajectory."
            )
        X_projected, _ = self.extract_traj(sol['x'])
        if return_solution:
            return X_projected[:, 0].full().reshape(4), sol
        return X_projected[:, 0].full().reshape(4)

    def plot_linearized_hit_and_run_samples(
        self,
        samples,
        linearization_samples=None,
        projected_mask=None,
        seeds=None,
        x_max=None,
        view_3d=False,
        show=True,
    ):
        """Plot all pairwise state projections of Hit-and-Run samples.

        Feasible seeds, linearized Hit-and-Run proposals, and their nonlinear
        projections are drawn separately, following the visual structure of
        the discontinuous-control sampling visualization.
        """
        samples = self._normalize_state_batch(samples)
        if linearization_samples is None:
            linearization_samples = samples.copy()
        else:
            linearization_samples = self._normalize_state_batch(
                linearization_samples
            )
            if linearization_samples.shape[0] != samples.shape[0]:
                raise ValueError(
                    "linearization_samples must contain one proposal per sample."
                )
        if projected_mask is None:
            projected_mask = np.zeros(samples.shape[0], dtype=bool)
        else:
            projected_mask = np.asarray(projected_mask, dtype=bool).reshape(-1)
            if projected_mask.size != samples.shape[0]:
                raise ValueError("projected_mask length must match the number of samples.")

        if seeds is not None:
            seeds = self._normalize_state_batch(seeds)

        state_labels = (
            "Position (m)",
            "Velocity (m/s)",
            "Angle (rad)",
            "Angular velocity (rad/s)",
        )
        state_bounds = np.array(
            [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound],
            dtype=float,
        )
        plot_bounds = (
            state_bounds
            if x_max is None
            else np.asarray(x_max, dtype=float).reshape(-1)
        )
        if (
            plot_bounds.shape != (4,)
            or not np.all(np.isfinite(plot_bounds))
            or np.any(plot_bounds <= 0.0)
        ):
            raise ValueError("x_max must contain four positive finite values.")

        if view_3d:
            fig = plt.figure(figsize=(10, 8), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")

            for proposal, projection in zip(linearization_samples, samples):
                ax.plot(
                    [proposal[0], projection[0]],
                    [proposal[1], projection[1]],
                    [proposal[2], projection[2]],
                    color="0.75",
                    alpha=0.45,
                    linewidth=0.65,
                    zorder=0,
                )

            if linearization_samples.size:
                ax.scatter(
                    linearization_samples[:, 0],
                    linearization_samples[:, 1],
                    linearization_samples[:, 2],
                    s=28,
                    marker="x",
                    color="0.5",
                    linewidths=1.0,
                    alpha=0.7,
                    label="Linearized HR proposal",
                    depthshade=False,
                )

            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                samples[:, 2],
                color="tab:blue",
                s=48,
                alpha=0.9,
                edgecolors="none",
                label="Projected feasible sample",
                depthshade=False,
            )

            if seeds is not None:
                ax.scatter(
                    seeds[:, 0],
                    seeds[:, 1],
                    seeds[:, 2],
                    s=145,
                    marker="*",
                    color="black",
                    edgecolors="white",
                    linewidths=0.7,
                    label="Feasible seed",
                    depthshade=False,
                )

            ax.set_xlim(-1.05 * plot_bounds[0], 1.05 * plot_bounds[0])
            ax.set_ylim(-1.05 * plot_bounds[1], 1.05 * plot_bounds[1])
            ax.set_zlim(-1.05 * plot_bounds[2], 1.05 * plot_bounds[2])
            ax.set_xlabel(state_labels[0], labelpad=9)
            ax.set_ylabel(state_labels[1], labelpad=9)
            ax.set_zlabel(state_labels[2], labelpad=9)
            ax.view_init(elev=24, azim=-58)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", framealpha=0.95)
            ax.set_title(
                "Cart-pole Hit-and-Run: position, velocity, and angle"
            )
            if show:
                plt.show()
            return fig, ax

        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        fig, axes = plt.subplots(
            2, 3, figsize=(16, 9), constrained_layout=True
        )

        for ax, (state_i, state_j) in zip(axes.ravel(), pairs):
            # Draw proposal-to-projection displacement first so markers remain
            # legible above the connecting segments.
            for proposal, projection in zip(
                linearization_samples,
                samples,
            ):
                ax.plot(
                    [proposal[state_i], projection[state_i]],
                    [proposal[state_j], projection[state_j]],
                    color="0.75",
                    alpha=0.45,
                    linewidth=0.6,
                    zorder=0,
                )

            if linearization_samples.size:
                ax.scatter(
                    linearization_samples[:, state_i],
                    linearization_samples[:, state_j],
                    s=24,
                    marker="x",
                    color="0.55",
                    linewidths=1.0,
                    alpha=0.65,
                    label="Linearized HR proposal",
                    zorder=1,
                )

            ax.scatter(
                samples[:, state_i],
                samples[:, state_j],
                color="tab:blue",
                s=42,
                alpha=0.88,
                edgecolors="none",
                label="Projected feasible sample",
                zorder=2,
            )
            if seeds is not None:
                ax.scatter(
                    seeds[:, state_i],
                    seeds[:, state_j],
                    s=120,
                    marker="*",
                    color="black",
                    edgecolors="white",
                    linewidths=0.65,
                    label="Feasible seed",
                    zorder=3,
                )
            ax.axvline(
                -plot_bounds[state_i], color="0.35", linestyle="--",
                linewidth=0.7, alpha=0.45,
            )
            ax.axvline(
                plot_bounds[state_i], color="0.35", linestyle="--",
                linewidth=0.7, alpha=0.45,
            )
            ax.axhline(
                -plot_bounds[state_j], color="0.35", linestyle="--",
                linewidth=0.7, alpha=0.45,
            )
            ax.axhline(
                plot_bounds[state_j], color="0.35", linestyle="--",
                linewidth=0.7, alpha=0.45,
            )
            ax.set_xlim(-1.05 * plot_bounds[state_i], 1.05 * plot_bounds[state_i])
            ax.set_ylim(-1.05 * plot_bounds[state_j], 1.05 * plot_bounds[state_j])
            ax.set_xlabel(state_labels[state_i])
            ax.set_ylabel(state_labels[state_j])
            ax.grid(True, alpha=0.25)

        axes[0, 0].legend(loc="best", framealpha=0.95)
        fig.suptitle(
            "Cart-pole Hit-and-Run: feasible seeds, linearized proposals, "
            "and nonlinear projections"
        )
        if show:
            plt.show()
        return fig, axes
    
    @staticmethod
    def _validate_projection_weight(Qx):
        """Return a symmetric positive-semidefinite 2-by-2 state weight."""
        if Qx is None:
            Qx = np.eye(4)
        elif np.isscalar(Qx):
            Qx = float(Qx) * np.eye(4)
        else:
            Qx = np.asarray(Qx, dtype=float)
            if Qx.shape == (4,):
                Qx = np.diag(Qx)
        Qx = np.asarray(Qx, dtype=float)
        if Qx.shape != (4, 4) or not np.all(np.isfinite(Qx)):
            raise ValueError("Qx must be a finite scalar, length-4 vector, or 4-by-4 matrix.")
        if not np.allclose(Qx, Qx.T, atol=1e-12, rtol=0.0):
            raise ValueError("Qx must be symmetric.")
        if np.min(np.linalg.eigvalsh(Qx)) < -1e-12:
            raise ValueError("Qx must be positive semidefinite.")
        return 0.5 * (Qx + Qx.T)

    def collect_linearized_hit_and_run_samples(
        self,
        num_samples=100,
        num_feasible_seeds=5,
        max_seed_attempts=1000,
        projection_Qx=None,
        project_infeasible=True,
        use_bilevel=False,
        reference_method=None,
        slack_penalty="quadratic",
        close_to_bounds=True,
        random_seed=None,
        x_max=None,
        plot_results=True,
        plot_3d=False,
        show=True,
        warm_start=True,
    ):
        """Collect nonlinear-feasible states with local LMPC Hit-and-Run.

        Reference solutions are obtained by ``reference_method``. Each
        reference trajectory defines a time-varying affine OCP. Bounded
        Hit-and-Run then generates exactly ``num_samples / num_feasible_seeds``
        candidates per reference. Every candidate is used as the target of the
        nonlinear projection problem; the projection solution supplies both
        the returned feasible state and its locally optimal NMPC trajectory.

        See https://arxiv.org/abs/2604.09118 for the linear MPC boundary LP.
        
        Args:
            num_samples (int): Total number of nonlinear-feasible samples to collect.
            num_feasible_seeds (int): Number of independent nonlinear-feasible seeds to use.
            max_seed_attempts (int): Maximum uniform draws to find feasible seeds.
            projection_Qx (scalar, vector, or matrix): Optional positive-semidefinite weight for the Euclidean-distance projection. 
            project_infeasible (bool): Deprecated compatibility flag. It must
                remain True because every candidate is projected.
            use_bilevel (bool): Legacy alias selecting bilevel references when
                ``reference_method`` is not supplied.
            reference_method (str or None): ``"bilevel"``, ``"slacked"``, or
                ``"uniform"`` reference generation.
            If None, the identity is used.
            random_seed (int or None): Optional seed for the random number generator.
            warm_start (bool): Reuse the previous successful bilevel solution
                and the previous projected trajectory in each Hit-and-Run chain.
            x_max (array-like or None): Optional 4-element array of positive state bounds for the Hit-and-Run LPs. 
            If None, the nonlinear MPC bounds are used.
            plot_results (bool): If True, plot the collected samples and seeds.
            plot_3d (bool): If True, show position, velocity, and angle in one
                3D view instead of all six pairwise projections.
            show (bool): If True, display the plot immediately. If False, return the figure and axes for further customization.
        """
        # Validate the input arguments.
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_feasible_seeds <= 0:
            raise ValueError("num_feasible_seeds must be positive.")
        if max_seed_attempts <= 0:
            raise ValueError("max_seed_attempts must be positive.")
        if not project_infeasible:
            raise ValueError(
                "The paper-compliant pipeline projects every linearized candidate; "
                "project_infeasible=False is no longer supported."
            )
        if num_samples == 0:
            self.last_linearized_sampling_stats = {"requested_samples": 0, "returned_samples": 0}
            return np.empty((0, 4))
        if num_feasible_seeds > num_samples:
            raise ValueError("num_feasible_seeds cannot exceed num_samples.")
        
        # The projection weight is validated here, but not used until the nonlinear projection step
        Qx = self._validate_projection_weight(projection_Qx)

        # State bound definition
        model_bounds = np.array([self.p_bound, self.v_bound, self.theta_bound, self.omega_bound], dtype=float)
        state_bounds = model_bounds if x_max is None else np.asarray(x_max, dtype=float)
        if state_bounds.shape != (4,) or np.any(state_bounds <= 0.0):
            raise ValueError("x_max must contain four positive state bounds.")
        if np.any(state_bounds > model_bounds + 1e-12):
            raise ValueError("x_max cannot exceed the nonlinear MPC state bounds.")

        if reference_method is None:
            reference_method = "bilevel" if use_bilevel else "uniform"
        reference_method = str(reference_method).lower()
        if reference_method not in {"uniform", "bilevel", "slacked"}:
            raise ValueError(
                "reference_method must be 'uniform', 'bilevel', or 'slacked'."
            )

        # Generate the reference solutions used to build the local linearized OCPs.
        rng = np.random.default_rng(random_seed)
        target_seed_count = num_feasible_seeds
        if num_samples % target_seed_count != 0:
            raise ValueError(
                "num_samples must be divisible by num_feasible_seeds so each "
                "reference produces the same number M of candidates."
            )
        samples_per_reference = num_samples // target_seed_count
        reference_stats = {}
        reference_time_s = 0.0
        if reference_method == "bilevel":
            reference_start = time.perf_counter()
            seeds, seed_solutions = self.collect_feasible_samples_with_bilevel(
                x_max=state_bounds,
                num_samples=target_seed_count,
                max_attempts=max_seed_attempts,
                return_solutions=True,
                warm_start=warm_start,
            )
            reference_time_s = time.perf_counter() - reference_start
            seed_attempts = int(
                self.last_feasible_sampling_stats.get("attempts", max_seed_attempts)
            )
            reference_stats = dict(self.last_feasible_sampling_stats)
        elif reference_method == "slacked":
            reference_start = time.perf_counter()
            seeds, seed_solutions = self.collect_feasible_samples_with_slacked(
                x_max=state_bounds,
                close_to_bounds=close_to_bounds,
                num_samples=target_seed_count,
                max_attempts=max_seed_attempts,
                plot_results=False,
                slack_penalty=slack_penalty,
                return_solutions=True,
            )
            reference_time_s = time.perf_counter() - reference_start
            seed_attempts = int(
                self.last_feasible_sampling_stats.get("attempts", max_seed_attempts)
            )
            reference_stats = dict(self.last_feasible_sampling_stats)
        else:
            seeds = []
            seed_solutions = []
            seed_attempts = 0
            while len(seeds) < target_seed_count and seed_attempts < max_seed_attempts:
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
                    # Store the feasible seed and its corresponding solution
                    seeds.append(candidate)
                    seed_solutions.append(solution)
        if len(seeds) != target_seed_count:
            raise RuntimeError(
                f"Collected {len(seeds)}/{target_seed_count} required "
                f"{reference_method} reference solutions."
            )

        chains = []
        for reference_index, (seed, solution) in enumerate(
            zip(seeds, seed_solutions)
        ):
            linear_ocp = self._linearized_ocp_at_solution(solution)
            box_margin = float(np.min(state_bounds - np.abs(seed)))
            ocp_margin = float(np.min(linear_ocp["b"]))
            if box_margin < -BOUNDARY_LP_FEASIBILITY_TOL:
                raise RuntimeError(
                    f"Reference {reference_index} is outside the Hit-and-Run "
                    f"initial-state box by {-box_margin:.3e}."
                )
            chains.append({
                "reference_index": reference_index,
                "seed": seed.copy(),
                "seed_solution": solution,
                "delta_x": np.zeros(4),
                "linear_ocp": linear_ocp,
                "origin_box_margin": box_margin,
                "origin_ocp_margin": ocp_margin,
                "steps": 0,
                "generated": 0,
                "boundary_lp_failures": 0,
                "projection_solution": solution,
            })

        # Accepted samples and raw linearized-MPC candidates are stored
        # separately. The latter make rejection/projection rates auditable.
        samples = []
        controls = []
        accepted_projected = []
        raw_candidates = []
        accepted_raw_candidates = []
        projection_needed = 0
        projection_failures = 0
        boundary_lp_failures = 0
        chain_index = 0
        # Failed projections or numerically bad directions must not create an
        # unbounded loop. This allowance is deliberately generous.
        max_chain_updates = 10 * num_samples
        chain_updates = 0
        candidates_generated = 0
        boundary_lp_time_s = 0.0
        projection_time_s = 0.0

        # Round-robin Hit-and-Run updates across the local linear models.
        while candidates_generated < num_samples and chain_updates < max_chain_updates:
            chain = chains[chain_index % len(chains)]
            chain_index += 1
            if chain["generated"] >= samples_per_reference:
                continue

            # Normalizing an isotropic Gaussian vector gives a direction that
            # is uniform on the unit sphere.
            direction = rng.normal(size=4)
            direction /= np.linalg.norm(direction)

            # Two LPs identify the positive and negative endpoints of the
            # feasible chord through the current chain state.
            lp_start = time.perf_counter()
            alpha_plus = self._linearized_boundary_step(
                chain["delta_x"],
                direction,
                chain["linear_ocp"],
                state_bounds,
            )
            alpha_minus = self._linearized_boundary_step(
                chain["delta_x"],
                -direction,
                chain["linear_ocp"],
                state_bounds,
            )
            boundary_lp_time_s += time.perf_counter() - lp_start
            chain_updates += 1
            if alpha_plus is None or alpha_minus is None:
                boundary_lp_failures += 1
                chain["boundary_lp_failures"] += 1
                at_origin = np.linalg.norm(
                    chain["delta_x"], ord=np.inf
                ) <= BOUNDARY_LP_FEASIBILITY_TOL
                if at_origin:
                    raise RuntimeError(
                        "Boundary LP failed at the guaranteed-feasible origin "
                        f"of reference {chain['reference_index']}; "
                        f"box margin={chain['origin_box_margin']:.3e}, "
                        f"OCP margin={chain['origin_ocp_margin']:.3e}. "
                        "This indicates a solver/interface failure rather than "
                        "a failed Hit-and-Run direction."
                    )
                chain["delta_x"] = np.zeros(4)
                continue
            if alpha_plus < 0.0 or alpha_minus < 0.0:
                # Defensive guard against a solver returning an invalid chord.
                boundary_lp_failures += 1
                chain["delta_x"] = np.zeros(4)
                continue

            # Stay infinitesimally inside the chord. This prevents accumulated
            # LP feasibility tolerances from moving the next chain state just
            # outside its local polytope.
            chord_width = alpha_plus + alpha_minus
            # Stay farther inside than the LP feasibility tolerance; 1e-10 was
            # too small to prevent the next chain point becoming numerically
            # infeasible after an endpoint solve.
            margin = min(
                10.0 * BOUNDARY_LP_FEASIBILITY_TOL,
                0.25 * chord_width,
            )
            beta = rng.uniform(-alpha_minus + margin, alpha_plus - margin)
            chain["delta_x"] = chain["delta_x"] + beta * direction
            chain["steps"] += 1

            # Compute the candidate state by adding the perturbation to the
            # reference, then always use it as the nonlinear projection target.
            candidate = chain["seed"] + chain["delta_x"]
            raw_candidates.append(candidate.copy())
            chain["generated"] += 1
            candidates_generated += 1
            projection_start = time.perf_counter()
            try:
                # Projection changes only the returned training sample.
                # The HR chain remains at its linearly feasible candidate,
                # since a nonlinear projection need not be feasible for
                # the fixed local affine OCP.
                projected_state, projected_solution = self.project_onto_feasible_set(
                    candidate,
                    Qx=Qx,
                    feasible_reference=chain["seed"],
                    reference_solution=(
                        chain["projection_solution"]
                        if warm_start else chain["seed_solution"]
                    ),
                    return_solution=True
                )
            except RuntimeError:
                projection_time_s += time.perf_counter() - projection_start
                projection_failures += 1
                continue
            projection_time_s += time.perf_counter() - projection_start
            # Record whether the projection changed the target.
            projected_was_needed = not np.allclose(
                projected_state, candidate, atol=1e-6, rtol=0.0
            )
            if projected_was_needed:
                projection_needed += 1
            _, projected_U = self.extract_traj(projected_solution["x"])
            accepted_raw_candidates.append(candidate.copy())
            samples.append(projected_state)
            if warm_start:
                chain["projection_solution"] = projected_solution
            controls.append(float(projected_U[0]))
            accepted_projected.append(projected_was_needed)

        # Save the sampling statistics for later inspection, and return the collected samples.
        result = np.asarray(samples, dtype=float).reshape(-1, 4)
        self.last_feasible_sample_controls = np.asarray(controls, dtype=float)
        self.last_linearized_sampling_stats = {
            "requested_samples": int(num_samples),
            "returned_samples": int(result.shape[0]),
            "seed_attempts": int(seed_attempts),
            "feasible_seeds": np.asarray(seeds),
            "raw_candidates": np.asarray(raw_candidates).reshape(-1, 4),
            "accepted_raw_candidates": np.asarray(
                accepted_raw_candidates, dtype=float
            ).reshape(-1, 4),
            "accepted_projected": np.asarray(accepted_projected, dtype=bool),
            "projection_needed": int(projection_needed),
            "projection_failures": int(projection_failures),
            "boundary_lp_failures": int(boundary_lp_failures),
            "boundary_lp_failures_per_reference": np.asarray(
                [chain["boundary_lp_failures"] for chain in chains],
                dtype=int,
            ),
            "chain_updates": int(chain_updates),
            "projection_Qx": Qx.copy(),
            "warm_start": bool(warm_start),
            "reference_method": reference_method,
            "reference_statistics": reference_stats,
            "samples_per_reference": int(samples_per_reference),
            "reference_time_s": float(reference_time_s),
            "boundary_lp_time_s": float(boundary_lp_time_s),
            "projection_time_s": float(projection_time_s),
        }
        if candidates_generated < num_samples:
            raise RuntimeError(
                f"Generated only {candidates_generated}/{num_samples} candidates after "
                f"{chain_updates} Hit-and-Run updates. Inspect "
                "last_linearized_sampling_stats for failure counts."
            )
        if result.shape[0] < num_samples:
            raise RuntimeError(
                f"Projected only {result.shape[0]}/{num_samples} candidates; "
                "the Np*M projection pipeline is incomplete. Inspect "
                "last_linearized_sampling_stats for failure counts."
            )
        if plot_results:
            self.last_linearized_sampling_plot = (
                self.plot_linearized_hit_and_run_samples(
                    result,
                    linearization_samples=accepted_raw_candidates,
                    projected_mask=accepted_projected,
                    seeds=seeds,
                    x_max=state_bounds,
                    view_3d=plot_3d,
                    show=show,
                )
            )
        return result

    def is_feasible(self, x0, Nsim=None, tol=1e-3, max_steps=50):
        """Return True if the MPC problem is feasible for initial state x0."""
        try:
            self.solver(
                x0=self.w0,
                lbx=self.lbw,
                ubx=self.ubw,
                lbg=self.lbg,
                ubg=self.ubg,
                p=x0,
            )
        except Exception:
            return False
        return bool(self.solver.stats().get("success", False))

    def plot_feasible_set(
        self,
        state_i=0,
        state_j=2,
        samples=None,
        feasible_mask=None,
        check_mpc=False,
        color_initial_state_regions=False,
        small_omega_threshold=1.0,
        swing_up_theta_tolerance=np.pi / 6,
        show=True,
        ax=None,
    ):
        """Visualize the state bounds as a feasible region in 2D."""
        bounds = np.array(
            [self.p_bound, self.v_bound, self.theta_bound, self.omega_bound],
            dtype=float,
        )
        if bounds.size < max(state_i, state_j) + 1:
            raise ValueError("state_i/state_j out of range for bounds.")
        if state_i == state_j:
            raise ValueError("state_i and state_j must be different indices.")

        x_min = -bounds[state_i]
        x_max = bounds[state_i]
        y_min = -bounds[state_j]
        y_max = bounds[state_j]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="b",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        if samples is not None:
            samples_arr = np.asarray(samples, dtype=float)
            if samples_arr.ndim == 3:
                samples_arr = samples_arr.reshape(-1, samples_arr.shape[-1])
            if samples_arr.ndim == 2 and samples_arr.shape[0] == 4:
                samples_arr = samples_arr.T
            if samples_arr.ndim != 2 or samples_arr.shape[1] != 4:
                raise ValueError("samples must have shape (N, 4) or (4, N).")

            if feasible_mask is None and check_mpc:
                feasible_mask = np.array([self.is_feasible(x0) for x0 in samples_arr], dtype=bool)
            elif feasible_mask is None:
                feasible_mask = np.all(np.abs(samples_arr) <= bounds, axis=1)
            else:
                feasible_mask = np.asarray(feasible_mask, dtype=bool)
                if feasible_mask.shape[0] != samples_arr.shape[0]:
                    raise ValueError("feasible_mask length must match number of samples.")

            print(f"Feasible samples: {np.sum(feasible_mask)} / {len(samples_arr)}")

            if color_initial_state_regions:
                theta = samples_arr[:, 2]
                omega = samples_arr[:, 3]
                small_omega = np.abs(omega) <= small_omega_threshold
                stabilization_mask = feasible_mask & (theta >= -np.pi / 3) & (theta <= np.pi / 3) & small_omega
                swing_up_mask = feasible_mask & (np.abs(np.abs(theta) - np.pi) <= swing_up_theta_tolerance) & small_omega
                other_feasible_mask = feasible_mask & ~stabilization_mask & ~swing_up_mask

                if np.any(other_feasible_mask):
                    ax.scatter(
                        samples_arr[other_feasible_mask, state_i],
                        samples_arr[other_feasible_mask, state_j],
                        s=20,
                        color="tab:green",
                        alpha=0.45,
                        label="Other feasible samples",
                    )
                if np.any(stabilization_mask):
                    ax.scatter(
                        samples_arr[stabilization_mask, state_i],
                        samples_arr[stabilization_mask, state_j],
                        s=32,
                        color="tab:blue",
                        alpha=0.8,
                        label=r"Stabilization: $\theta \in [-\pi/3, \pi/3]$, small $\omega$",
                    )
                if np.any(swing_up_mask):
                    ax.scatter(
                        samples_arr[swing_up_mask, state_i],
                        samples_arr[swing_up_mask, state_j],
                        s=32,
                        color="tab:orange",
                        alpha=0.8,
                        label=r"Swing-up: $\theta \approx \pm\pi$, small $\omega$",
                    )
            else:
                ax.scatter(
                    samples_arr[feasible_mask, state_i],
                    samples_arr[feasible_mask, state_j],
                    s=20,
                    color="tab:green",
                    alpha=0.6,
                    label="Feasible samples",
                )
            if np.any(~feasible_mask):
                ax.scatter(
                    samples_arr[~feasible_mask, state_i],
                    samples_arr[~feasible_mask, state_j],
                    s=20,
                    color="tab:red",
                    alpha=0.6,
                    label="Infeasible samples",
                )

        ax.set_xlabel(f"x[{state_i}]")
        ax.set_ylabel(f"x[{state_j}]")
        ax.set_title("Feasible Set (State Bounds)")
        ax.set_xlim(x_min * 1.05, x_max * 1.05)
        ax.set_ylim(y_min * 1.05, y_max * 1.05)
        ax.grid(True, alpha=0.3)
        if samples is not None:
            ax.legend(loc="best")

        if show:
            plt.tight_layout()
            plt.show()

        return samples_arr[feasible_mask, :]
    
    def display_u0_bound_activity(self, u0_values, tol=1e-6, show=True, ax=None):
        """Plot how many collected first controls are active at the input bounds."""
        u0_arr = np.asarray(u0_values, dtype=float).reshape(-1)
        upper_active = np.isclose(u0_arr, self.u_bound, atol=tol, rtol=0.0)
        lower_active = np.isclose(u0_arr, -self.u_bound, atol=tol, rtol=0.0)
        any_active = upper_active | lower_active

        stats = {
            "total": int(u0_arr.size),
            "lower_active": int(np.count_nonzero(lower_active)),
            "upper_active": int(np.count_nonzero(upper_active)),
            "any_active": int(np.count_nonzero(any_active)),
            "inactive": int(np.count_nonzero(~any_active)),
            "tol": float(tol),
            "u_bound": float(self.u_bound),
        }
        self.last_u0_bound_activity = stats

        print(
            "u_0 bound-active samples: "
            f"{stats['any_active']}/{stats['total']} "
            f"(lower: {stats['lower_active']}, upper: {stats['upper_active']}, tol={tol:g})"
        )

        # if ax is None:
        #     _, ax = plt.subplots(figsize=(8, 4.5))

        # bins = min(max(10, int(np.sqrt(max(u0_arr.size, 1)))), 50)
        # ax.hist(u0_arr, bins=bins, color="tab:blue", alpha=0.72, edgecolor="white")
        # ax.axvline(-self.u_bound, color="tab:red", linestyle="--", linewidth=1.5, label="lower bound")
        # ax.axvline(self.u_bound, color="tab:red", linestyle="--", linewidth=1.5, label="upper bound")
        # ax.scatter(
        #     u0_arr[any_active],
        #     np.zeros(stats["any_active"]),
        #     color="tab:red",
        #     marker="|",
        #     s=130,
        #     linewidths=1.4,
        #     label="bound-active samples",
        #     zorder=3,
        # )
        # ax.set_title(
        #     f"u_0 Bound Activity: {stats['any_active']}/{stats['total']} active "
        #     f"({stats['lower_active']} lower, {stats['upper_active']} upper)"
        # )
        # ax.set_xlabel("u_0")
        # ax.set_ylabel("Sample count")
        # ax.grid(True, alpha=0.3)
        # ax.legend(loc="best")

        # if show:
        #     plt.tight_layout()
        #     plt.show()

        return stats

    def plot_policy_projections(self, samples, u0_values, show=True):
        """Plot the sampled MPC policy against each cart-pole state component."""
        samples_arr = np.asarray(samples, dtype=float)
        if samples_arr.ndim == 2 and samples_arr.shape[1] != 4 and samples_arr.shape[0] == 4:
            samples_arr = samples_arr.T
        if samples_arr.ndim != 2 or samples_arr.shape[1] != 4:
            raise ValueError("samples must have shape (N, 4) or (4, N).")

        u0_arr = np.asarray(u0_values, dtype=float).reshape(-1)
        if u0_arr.size != samples_arr.shape[0]:
            raise ValueError("u0_values length must match number of samples.")

        state_labels = [
            "Position (m)",
            "Velocity (m/s)",
            "Angle (rad)",
            "Angular velocity (rad/s)",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = axes.ravel()

        for state_idx, ax in enumerate(axes):
            ax.scatter(
                samples_arr[:, state_idx],
                u0_arr,
                s=28,
                color="tab:blue",
                alpha=0.75,
                edgecolors="none",
            )
            ax.axhline(self.u_bound, color="tab:red", linestyle="--", linewidth=1.0)
            ax.axhline(-self.u_bound, color="tab:red", linestyle="--", linewidth=1.0)
            ax.set_xlabel(state_labels[state_idx])
            ax.set_ylabel(r"First MPC action $u_0$")
            ax.grid(True, alpha=0.3)

        fig.suptitle(r"Sampled MPC policy projections: $u_0 = \pi(x_0)$")
        fig.tight_layout()
        if show:
            plt.show()

        return fig, axes

    def collect_feasible_samples_with_slacked(
        self,
        x_max=None,
        close_to_bounds=False,
        u0_active_bound=False,
        num_samples=100,
        max_attempts=1000,
        plot_results=True,
        slack_penalty="quadratic",
        print_stationarity_perturbation=False,
        print_optimality=False,
        return_solutions=False,
    ):
        """Collect feasible states using the paper's slacked NMPC formulation.

        The retained ``slack_penalty`` argument must be ``"quadratic"``.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative.")
        if str(slack_penalty).lower() != "quadratic":
            raise ValueError(
                "The paper-compliant slacked formulation only supports "
                "slack_penalty='quadratic'."
            )
        if x_max is None:
            x_max = np.array([self.p_bound, self.v_bound, self.theta_bound, self.omega_bound], dtype=float)
        else:
            x_max = np.asarray(x_max, dtype=float)
            if x_max.shape != (4,):
                raise ValueError("x_max must have shape (4,).")

        samples = []
        solutions = []
        u0_values = []
        attempts_run = 0
        successful_extreme_solves = 0
        feasible_candidates = 0
        optimal_slacked_solutions = 0

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
                print_stationarity_perturbation=print_stationarity_perturbation,
                return_traj=True
            )
            
            if return_status != "Solve_Succeeded":
                print(f"Attempt {attempt + 1}: solver failed (status: {return_status}), skipping...")
                continue
            
            if print_optimality:
                # Solve a new MPC problem with the found x0_opt. Then, return all the states and controls in the trajectory
                _ = self.solve_MPC(x0_opt, warm_start=w_opt_slacked, return_traj=True, check_optimality=True)                

            successful_extreme_solves += 1
            if self.last_slacked_c_star_is_zero:
                optimal_slacked_solutions += 1
            # if not self.is_feasible(x0_opt):
            #     print(f"Attempt {attempt + 1}: x0 not feasible, skipping...")
            #     continue
            feasible_candidates += 1

            samples.append(x0_opt)
            solutions.append({"x": ca.DM(w_opt_slacked)})
            u0_values.append(u0_opt)
            print(f"Collected {len(samples)}/{num_samples}")

        if len(samples) < num_samples:
            print(f"Only collected {len(samples)} samples after {max_attempts} attempts")
        else:
            print(f"Collected {num_samples} samples")
        print(
            "Original-MPC-compatible slacked solutions (c*=0): "
            f"{optimal_slacked_solutions}/{successful_extreme_solves}"
        )
        
        def all_unique(samples):
            return len(samples) == len({
                (x.shape, x.dtype.str, x.tobytes()) for x in samples
            })
        
        if all_unique(samples):
            print("All collected samples are unique.")
        else:
            print("Warning: Some collected samples are duplicates.")
        
        x0_samples = np.asarray(samples, dtype=float).reshape(-1, 4)
        u0_values = np.array(u0_values, dtype=float)
        self.last_feasible_sample_controls = u0_values
        self.last_feasible_sampling_stats = {
            "slack_penalty": str(slack_penalty).lower(),
            "requested_samples": int(num_samples),
            "returned_samples": int(x0_samples.shape[0]),
            "attempts": int(attempts_run),
            "successful_extreme_solves": int(successful_extreme_solves),
            "c_star_evaluated_solutions": int(successful_extreme_solves),
            "feasible_candidates": int(feasible_candidates),
            "optimal_slacked_solutions": int(optimal_slacked_solutions),
            "optimal_slacked_solution_rate": (
                float(optimal_slacked_solutions / successful_extreme_solves)
                if successful_extreme_solves else 0.0
            ),
            "c_star_relative_tolerance": 1e-6,
        }
        if plot_results:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            feasible_mask = np.array([self.is_feasible(x0) for x0 in x0_samples], dtype=bool)
            feas_samples = self.plot_feasible_set(
                state_i=0,
                state_j=2,
                samples=x0_samples,
                feasible_mask=feasible_mask,
                color_initial_state_regions=True,
                show=False,
                ax=axes[0],
            )
            self.plot_feasible_set(
                state_i=2,
                state_j=3,
                samples=x0_samples,
                feasible_mask=feasible_mask,
                color_initial_state_regions=True,
                show=False,
                ax=axes[1],
            )
            axes[0].set_title("Initial States: Position vs Theta")
            axes[1].set_title("Initial States: Theta vs Omega")
            fig.tight_layout()
            plt.show()
            self.plot_policy_projections(x0_samples, u0_values)
            return feas_samples
        
        if return_solutions:
            return x0_samples, solutions
        return x0_samples
    
    def collect_feasible_samples_with_bilevel(
        self,
        x_max,
        num_samples,
        max_attempts=100,
        plot_results=False,
        return_solutions=False,
        complementarity_method="ccopt",
        check_sosc=False,
        scholtes_relaxations=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
        warm_start=True,
    ):
        """Collect initial states found by successful solves with bilevel optimization.
        The collected samples are guaranteed to be feasible and optimal for the MPC problem.
        Repeated calls reuse the cached bilevel solver when their bounds and
        complementarity settings match.
        """
        samples = []
        solutions = []
        u0_values = []
        attempts_run = 0
        feasible_and_optimal_candidates = 0
        feasible_candidates = 0
        
        # Set up the bilevel optimization problem
        bilevel_solver, w0, lbw, ubw, lbg, ubg, slack_block_sizes, mpc_vars, lambda_vars, slack_vars = self.bilevel_active_set(
            x_max,
            return_solver=True,
            complementarity_method=complementarity_method,
            scholtes_relaxations=scholtes_relaxations,
        )
        bilevel_initial_guess = w0

        for attempt in range(max_attempts):
            if len(samples) >= num_samples:
                break
            attempts_run = attempt + 1
 
            # Sample slack penalty weights for the bilevel optimization
            # Compute the max numbers of spikes for the sampled slack penalty weights
            n_w = mpc_vars(ca.vertcat(*w0)).numel()
            m_h = lambda_vars(ca.vertcat(*w0))[0].numel()
            max_spikes = n_w - m_h
            # print(f"n_w = {n_w}, m_h = {m_h}, max_spikes = {max_spikes}")
            # raise NotImplementedError("The bilevel sampling method is not fully implemented yet.")
            rho_value, _ = self.spiky_residual_weights(
                        slack_block_sizes,
                        min_spikes=1,
                        max_spikes=max_spikes,
                        base_scale=1e-6,
                        spike_factor_range=(1e0, 1e2),
                        spike_probability=0.6,
                    )
            
            x0_value = np.array([0.0, 0.0, 0.0, 0.0])
            p_value = np.concatenate([x0_value, rho_value])
            # Solve the bilevel optimization problem with the sampled slack penalty weights
            sol = bilevel_solver(
                x0=bilevel_initial_guess,
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
            if warm_start:
                bilevel_initial_guess = sol['x']
            
            x0_feasible = sol['x'][0:4].full().flatten()
            
            mpc_solution = mpc_vars(sol['x']).full().flatten()
            u0_feasible = mpc_solution[4:5]
            lambda_opt, mu_opt = lambda_vars(sol['x'])
            slack_opt = slack_vars(sol['x'])
            
            if check_sosc:
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
                H = self.hessian_L(
                    mpc_solution, lambda_opt, mu_opt
                ).full()
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
                    # Match the solution-dictionary interface returned by the
                    # original MPC solver. Hit-and-Run uses this decision vector
                    # both to linearize the trajectory and to warm-start the
                    # nonlinear projection.
                    solutions.append({"x": ca.DM(mpc_solution)})
                    u0_values.append(u0_feasible)
            else:
                samples.append(x0_feasible)
                # Match the solution-dictionary interface returned by the
                # original MPC solver. Hit-and-Run uses this decision vector
                # both to linearize the trajectory and to warm-start the
                # nonlinear projection.
                solutions.append({"x": ca.DM(mpc_solution)})
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
            "complementarity_method": str(complementarity_method).lower(),
            "solver_cache_hit": bool(self._last_bilevel_solver_cache_hit),
            "warm_start": bool(warm_start),
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
            self.plot_policy_projections(x0_samples, u0_values)
        
        if return_solutions:
            return x0_samples, solutions
        return x0_samples
           
    def close_loop_simulation(self, x0, Nsim = 60, control_policy=None, plot_results=True):
        """Simulate the cart-pole system and plot the results.
        
        Parameters
        ----------
        x0 : array_like
            Initial state [position, velocity, angle, angular_velocity].
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step. If None, the MPC controller will be used.
        Nsim : int, optional
            Number of MPC/control intervals to simulate (default: 60). Each
            interval has length ``dt_mpc`` and is integrated with ``dt_sim``.
        plot_results : bool, optional
            Whether to plot the results (default: True).
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory at MPC/control interval boundaries with shape
            (Nsim+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (Nsim,).
        """
        sim_substeps = int(round(self.dt_mpc / self.dt_sim))
        if sim_substeps < 1 or not np.isclose(sim_substeps * self.dt_sim, self.dt_mpc):
            raise ValueError(
                "dt_mpc must be an integer multiple of dt_sim. "
                f"Got dt_mpc={self.dt_mpc} and dt_sim={self.dt_sim}."
            )

        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, 4))
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
            
            # Keep the input constant over one MPC/control interval while the
            # plant is integrated with the finer simulation step.
            x_current = x_traj[k].copy()
            for _ in range(sim_substeps):
                x_next = self.sim_step(x_current, [u_k])
                x_current = np.array(x_next.full()).flatten()
            x_traj[k + 1] = x_current
        
        if not plot_results:
            return x_traj, u_traj
        
        # Plot results
        time = np.arange(Nsim + 1) * self.dt_mpc
        time_u = np.arange(Nsim) * self.dt_mpc
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 10))
        title = "Cart-Pole Simulation"
        
        # Position
        axes[0].plot(time, x_traj[:, 0], 'b-', linewidth=2)
        axes[0].set_ylabel('Position (m)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        
        # Velocity
        axes[1].plot(time, x_traj[:, 1], 'r-', linewidth=2)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Angle
        axes[2].plot(time, x_traj[:, 2], 'g-', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (rad)', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # Angular velocity
        axes[3].plot(time, x_traj[:, 3], 'c-', linewidth=2)
        axes[3].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Control
        axes[4].step(time_u, u_traj, 'k-', where='post', linewidth=2)
        axes[4].set_ylabel('Control Force (N)', fontsize=10)
        axes[4].set_xlabel('Time (s)', fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj

    def run_closed_loop_batch(
        self,
        x0_batch,
        Nsim=60,
        control_policy=None,
        plot_results=True,
        skip_failed=True,
    ):
        """Run closed-loop simulations for a batch of initial states.

        Parameters
        ----------
        x0_batch : array_like
            Initial states with shape (n_batch, 4) or (4, n_batch).
        Nsim : int, optional
            Number of simulation steps for each trajectory.
        control_policy : callable, array_like, or sequence, optional
            Policy passed to ``close_loop_simulation``. If a sequence with one
            entry per initial state is provided, each entry is used for the
            corresponding trajectory.
        plot_results : bool, optional
            Whether to plot all successful trajectories.
        skip_failed : bool, optional
            If True, failed simulations are skipped. If False, the original
            exception is raised.

        Returns
        -------
        x_trajs : list[np.ndarray]
            State trajectories, each with shape (Nsim + 1, 4).
        u_trajs : list[np.ndarray]
            Control trajectories, each with shape (Nsim,).
        failed_indices : list[int]
            Indices of initial states that failed and were skipped.
        """
        x0_arr = np.asarray(x0_batch, dtype=float)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.reshape(1, -1)
        elif x0_arr.ndim != 2:
            raise ValueError("x0_batch must have shape (n_batch, 4) or (4, n_batch).")

        if x0_arr.shape[1] != 4 and x0_arr.shape[0] == 4:
            x0_arr = x0_arr.T
        if x0_arr.shape[1] != 4:
            raise ValueError("Each initial state must have length 4.")

        policy_is_batched = (
            control_policy is not None
            and not callable(control_policy)
            and hasattr(control_policy, "__len__")
            and len(control_policy) == x0_arr.shape[0]
        )

        x_trajs = []
        u_trajs = []
        failed_indices = []
        for i, x0 in enumerate(x0_arr):
            print(f"Simulating closed-loop trajectory {i + 1}/{x0_arr.shape[0]} for initial state: {x0}")
            policy_i = control_policy[i] if policy_is_batched else control_policy
            try:
                x_traj, u_traj = self.close_loop_simulation(
                    x0,
                    Nsim=Nsim,
                    control_policy=policy_i,
                    plot_results=False,
                )
            except Exception:
                if not skip_failed:
                    raise
                failed_indices.append(i)
                print(f"Skipping trajectory {i + 1} due to simulation failure.")
                continue

            if np.isnan(x_traj).any() or np.isnan(u_traj).any():
                if not skip_failed:
                    raise ValueError(f"Trajectory {i + 1} contains NaN values.")
                failed_indices.append(i)
                print(f"Skipping trajectory {i + 1} because it contains NaN values.")
                continue

            x_trajs.append(x_traj)
            u_trajs.append(u_traj)

        if plot_results and x_trajs:
            self.plot_closed_loop_batch(x_trajs, u_trajs)
        elif plot_results:
            print("No successful closed-loop trajectories to plot.")

        return x_trajs, u_trajs, failed_indices

    def plot_closed_loop_batch(self, x_trajs, u_trajs, show=True):
        """Plot a batch of cart-pole closed-loop trajectories."""
        if not x_trajs:
            raise ValueError("x_trajs is empty.")
        if len(x_trajs) != len(u_trajs):
            raise ValueError("x_trajs and u_trajs must have the same length.")

        time = np.arange(x_trajs[0].shape[0]) * self.dt_mpc
        time_u = np.arange(u_trajs[0].shape[0]) * self.dt_mpc

        fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
        state_labels = [
            "Position (m)",
            "Velocity (m/s)",
            "Angle (rad)",
            "Angular Velocity (rad/s)",
        ]

        for x_traj, u_traj in zip(x_trajs, u_trajs):
            for s in range(4):
                axes[s].plot(time, x_traj[:, s], alpha=0.5)
            axes[4].step(time_u, np.asarray(u_traj).flatten(), where="post", alpha=0.5)

        for s, label in enumerate(state_labels):
            if s in (0, 2):
                axes[s].axhline(y=0, color="k", linestyle="--", alpha=0.3)
            axes[s].set_ylabel(label, fontsize=10)
            axes[s].grid(True, alpha=0.3)

        axes[4].set_ylabel("Control Force (N)", fontsize=10)
        axes[4].set_xlabel("Time (s)", fontsize=10)
        axes[4].grid(True, alpha=0.3)

        fig.suptitle(f"Cart-Pole Closed-Loop Batch ({len(x_trajs)} trajectories)")
        plt.tight_layout()
        if show:
            plt.show()

        return fig, axes
    
    def animate(self, x0, control_policy, interval=50, save_path=None, title="Cart-Pole Animation"):
        """Animate the cart-pole system.
        
        Parameters
        ----------
        x0 : array_like
            Initial state [position, velocity, angle, angular_velocity].
        control_policy : callable or array_like
            Either a function that takes state and returns control u = policy(x),
            or an array of control inputs for each time step.
        N : int
            Number of simulation steps.
        interval : int, optional
            Time between frames in milliseconds (default: 50ms).
        save_path : str, optional
            If provided, save animation to this file path (e.g., 'animation.gif' or 'animation.mp4').
        title : str, optional
            Title for the animation.
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (N+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (N,).
        anim : matplotlib.animation.FuncAnimation
            The animation object.
        """
        sim_substeps = int(round(self.dt_mpc / self.dt_sim))
        if sim_substeps < 1 or not np.isclose(sim_substeps * self.dt_sim, self.dt_mpc):
            raise ValueError(
                "dt_mpc must be an integer multiple of dt_sim. "
                f"Got dt_mpc={self.dt_mpc} and dt_sim={self.dt_sim}."
            )

        # First, simulate to get the trajectory
        x_traj = np.zeros((self.N*3 + 1, 4))
        u_traj = np.zeros(self.N*3)
        
        x_traj[0] = np.array(x0).flatten()
        
        for k in range(self.N*3):
            if control_policy is None:
                u_k = self.solve_MPC(x_traj[k])  # Solve MPC to get control input for current state
            elif callable(control_policy):
                u_k = control_policy(x_traj[k])
            else:
                u_k = control_policy[k]
            
            if hasattr(u_k, '__len__'):
                u_k = float(u_k[0])
            else:
                u_k = float(u_k)
            
            u_traj[k] = u_k
            x_current = x_traj[k].copy()
            for _ in range(sim_substeps):
                x_next = self.sim_step(x_current, [u_k])
                x_current = np.array(x_next.full()).flatten()
            x_traj[k + 1] = x_current
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate axis limits based on trajectory
        x_min = np.min(x_traj[:, 0]) - 1.5
        x_max = np.max(x_traj[:, 0]) + 1.5
        y_min = -self.l - 0.5
        y_max = self.l + 0.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title(title)
        
        # Draw ground
        ax.plot([x_min, x_max], [0, 0], 'k-', linewidth=2)
        
        # Cart dimensions
        cart_width = 0.3
        cart_height = 0.2
        
        # Initialize objects
        cart = Rectangle((0, 0), cart_width, cart_height, 
                        fill=True, color='blue', ec='black', linewidth=2)
        ax.add_patch(cart)
        
        pole, = ax.plot([], [], 'r-', linewidth=4, marker='o', markersize=8, markerfacecolor='red')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        def init():
            """Initialize animation."""
            cart.set_xy((0, 0))
            pole.set_data([], [])
            time_text.set_text('')
            return cart, pole, time_text
        
        def update(frame):
            """Update animation frame."""
            # Get state at this frame
            pos = x_traj[frame, 0]
            theta = x_traj[frame, 2]
            
            # Update cart position
            cart.set_xy((pos - cart_width/2, 0))
            
            # Update pole position (pole rotates around cart center)
            pole_x = [pos, pos + self.l * np.sin(theta)]
            pole_y = [cart_height/2, cart_height/2 + self.l * np.cos(theta)]
            pole.set_data(pole_x, pole_y)
            
            # Update time text
            time = frame * self.dt_mpc
            time_text.set_text(f'Time: {time:.2f} s\nAngle: {theta:.3f} rad\nPosition: {pos:.3f} m')
            
            return cart, pole, time_text
        
        anim = FuncAnimation(fig, update, frames=self.N*3+1, init_func=init,
                           blit=True, interval=interval, repeat=True)
        
        plt.show()
        
        # Save animation if path is provided
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
            print(f"Animation saved!")
        
        return x_traj, u_traj, anim

if __name__ == "__main__":
    # Create cart-pole environment
    N = 50
    cart_pole = CartPole(N, dt_mpc=0.1, dt_sim=0.01, sym_type='SX')
    
    num_sims = 10
    # x0_batch = []
    # for _ in range(num_sims):
    #     # p0 = np.random.uniform(-cart_pole.p_bound, cart_pole.p_bound)
    #     theta0 = np.random.uniform(-cart_pole.theta_bound, cart_pole.theta_bound)
    #     x0_batch.append([-1, 0.0, theta0, 0.0])
    
    # cart_pole.plot_feasible_set(samples=training_initial_states, check_mpc=True)
    # x_max = np.array([2.0, 4.0, np.pi/3, 2.0], dtype=float)
    x_max = np.array([2.0, 5.0, np.pi, 10.0], dtype=float)
    # start_time = time.time()
    # x0_batch = cart_pole.collect_feasible_samples_with_slacked(
    #     x_max, 
    #     close_to_bounds=True, 
    #     u0_active_bound=False,
    #     num_samples=100, 
    #     max_attempts=1000, 
    #     plot_results=False,
    #     slack_penalty="quadratic",
    #     print_constraint_jacobian=False,
    #     print_optimality=True
    # )
    # print("\n")
    # print(f"Collected {len(x0_batch)} feasible samples in {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    # Compute an initial state using the bilevel optimization approach
    x0_batch = cart_pole.collect_feasible_samples_with_bilevel(
        x_max=x_max,
        num_samples=10,
        max_attempts=100,
        plot_results=False
    )
    print("\n")
    print(f"Collected {len(x0_batch)} feasible and optimal samples with bilevel optimization in {time.time() - start_time:.2f} seconds.")
    # x0_batch = cart_pole.collect_linearized_hit_and_run_samples(
    #     num_samples=1000,
    #     num_feasible_seeds=5,
    #     projection_Qx=100.0,
    #     use_bilevel=True,
    #     x_max=x_max,
    #     plot_results=True,
    #     plot_3d=True,
    # )
    
    # stats = cart_pole.last_linearized_sampling_stats
    # print("Seed attempts:", stats["seed_attempts"])
    # print("Projected:", stats["projected"].sum())
    # print("Projection failures:", stats["projection_failures"])
    
    # u0_values, fig, axes = cart_pole.solve_MPC_batch_first_controls(
    #     x0_batch,
    #     state_components=["theta"],
    #     x_axis_component="theta",
    #     plot_results=True,
    # )
    # Close-loop simulation with MPC control
    # cart_pole.run_closed_loop_batch(x0_batch, Nsim=cart_pole.N*4)
    # Animate
    # i = np.random.randint(0, len(x0_batch))
    # x_traj, u_traj, anim = cart_pole.animate(
    #     x0_batch[i], control_policy=None, 
    #     interval=50,  # 50ms between frames
    #     # save_path='cart_pole.gif'  # Optional: save to file
    # )
