"""Feasible-initial-state sampling for a planar point mass among obstacles.

The public sampling API mirrors the cart-pole model: an initial state is
feasible when its obstacle-avoiding double-integrator MPC reaches the fixed
goal with zero terminal velocity.  A reduced fixed-A-to-B parameterization is
retained internally to construct reliable feasible warm starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are
import matplotlib.pyplot as plt



@dataclass(frozen=True)
class CircularObstacle:
    center: Tuple[float, float]
    radius: float


def dense_obstacle_field() -> List[CircularObstacle]:
    """Return five staggered walls (30 circles) with alternating openings."""
    x_columns = (-3.2, -1.6, 0.0, 1.6, 3.2)
    y_rows = (-3.45, -2.30, -1.15, 0.0, 1.15, 2.30, 3.45)
    gap_rows = (1, 4, 2, 5, 3)
    return [
        CircularObstacle((x, y), 0.56)
        for x, gap in zip(x_columns, gap_rows)
        for row, y in enumerate(y_rows)
        if row != gap
    ]


class PointMass:
    """Planar double-integrator MPC and feasible-initial-state sampler.

    Public samples have the form ``[px, py, vx, vy]``.  Every feasibility
    check solves the same fixed-goal obstacle MPC used by ``solve_MPC``.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        dt: float = 0.35,
        dt_mpc: Optional[float] = None,
        dt_sim: Optional[float] = None,
        sym_type: str = "SX",
        start: Sequence[float] = (-4.6, -2.3),
        goal: Sequence[float] = (4.6, 0.0),
        workspace: Sequence[float] = (-5.0, 5.0, -4.0, 4.0),
        velocity_bound: float = 4.5,
        acceleration_bound: float = 5.0,
        body_radius: float = 0.10,
        obstacles: Optional[Iterable[CircularObstacle]] = None,
        collision_substeps: int = 2,
        horizon: Optional[int] = None,
    ) -> None:
        if horizon is None:
            horizon = 30 if N is None else N
        elif N is not None and int(N) != int(horizon):
            raise ValueError("N and horizon must match when both are provided.")
        if horizon < 6 or dt <= 0.0 or collision_substeps < 1:
            raise ValueError("Use horizon >= 6, dt > 0, and collision_substeps >= 1.")
        self.horizon = int(horizon)
        self.N = self.horizon
        self.dt_mpc = float(dt if dt_mpc is None else dt_mpc)
        self.dt_sim = float(self.dt_mpc if dt_sim is None else dt_sim)
        self.dt = self.dt_mpc
        self.sym_type = str(sym_type)
        self.start = np.asarray(start, dtype=float).reshape(2)
        self.goal = np.asarray(goal, dtype=float).reshape(2)
        self.workspace = np.asarray(workspace, dtype=float).reshape(4)
        self.velocity_bound = float(velocity_bound)
        self.acceleration_bound = float(acceleration_bound)
        self.body_radius = float(body_radius)
        self.obstacles = list(dense_obstacle_field() if obstacles is None else obstacles)
        self.collision_substeps = int(collision_substeps)

        self.dynamics = self.dynamics_func()
        self.mpc_step = self.step_func(self.dt_mpc)
        self.sim_step = self.step_func(self.dt_sim)
        self.step = self.mpc_step

        self.Q = np.eye(4)  # state cost matrix
        self.R = np.eye(2)  # control cost matrix
        self._linearized_boundary_solvers: Dict[Tuple[int, int], ca.Function] = {}
        self._extreme_solvers: Dict[str, Dict[str, object]] = {}
        self._projection_problem: Optional[Dict[str, object]] = None
        self.define_simple_MPC_control(self.N)
        
        self._build_reduced_parameterization()
        self._build_collision_maps()
        self._build_linear_constraints()
        # self._coordinate_bound_solver = self._build_coordinate_bound_solver()
        # self.coordinate_bounds = self._find_coordinate_bounds()
        self._feasible_seed: Optional[np.ndarray] = None
        self._slack_solvers: Dict[str, ca.Function] = {}
        
    def dynamics_func(self) -> ca.Function:
        """Return a CasADi function for the continuous double-integrator dynamics."""
        sym = getattr(ca, self.sym_type)
        state = sym.sym("state", 4)
        control = sym.sym("control", 2)
        x_dot = ca.vertcat(state[2], state[3], control[0], control[1])
        return ca.Function("point_mass_dynamics", [state, control], [x_dot])
    
    def step_func(self, dt: Optional[float] = None) -> ca.Function:
        """Return a CasADi function for the discrete double-integrator dynamics using RK4."""
        if dt is None:
            dt = self.dt_mpc
        x0 = getattr(ca, self.sym_type).sym('x0', 4)
        u0 = getattr(ca, self.sym_type).sym('u0', 2)
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

    @property
    def reduced_dimension(self) -> int:
        return int(self.null_basis.shape[1])
    
    def _symbolic_obstacle_margins(self, position: ca.SX) -> ca.SX:
        """Return symbolic squared-distance margins to all obstacles for a given position."""
        margins = []
        for obstacle in self.obstacles:
            delta = position - ca.DM(obstacle.center)
            margins.append(ca.sumsqr(delta) - (obstacle.radius + self.body_radius) ** 2)
        return ca.vertcat(*margins)

    def define_simple_MPC_control(self, N: Optional[int] = None) -> None:
        """Build the obstacle-avoiding, fixed-goal MPC with CasADi/IPOPT."""
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 4)  # state variable
        u = ca.SX.sym('u', 2)  # control variable
        x0 = ca.SX.sym('x0', 4)  # initial state parameter
        
        X = ca.SX.sym('X', 4, N+1)  # state trajectory
        U = ca.SX.sym('U', 2, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        x_next = self.mpc_step(x, u)
        xr = ca.DM(self.goal.tolist() + [0.0, 0.0])  # reference state (goal with zero velocity)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        # State bounds 
        self.state_lower = np.array([
            self.workspace[0] + self.body_radius,
            self.workspace[2] + self.body_radius,
            -self.velocity_bound,
            -self.velocity_bound,
        ])
        self.state_upper = np.array([
            self.workspace[1] - self.body_radius,
            self.workspace[3] - self.body_radius,
            self.velocity_bound,
            self.velocity_bound,
        ])
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', 4)  # initial state variable
        w += [Xk]
        self.lbw += self.state_lower.tolist()
        self.ubw += self.state_upper.tolist()
        self.w0 += [0.0 for _ in range(4)]
        g += [Xk - x0]  # initial state constraint
        self.lbg += [0.0 for _ in range(4)]
        self.ubg += [0.0 for _ in range(4)]
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 2)
            w += [uk]
            self.lbw += [-self.acceleration_bound] * 2  # control limits
            self.ubw += [self.acceleration_bound] * 2
            self.w0 += [0.0 for _ in range(2)]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            # Add obstacle avoidance constraints at each substep
            for substep in range(self.collision_substeps):
                tau = substep * self.dt_mpc / self.collision_substeps
                position = Xk[:2] + tau * Xk[2:] + 0.5 * tau**2 * uk
                obstacle_margin = self._symbolic_obstacle_margins(position)
                g += [obstacle_margin]
                self.lbg += [0.0 for _ in range(len(self.obstacles))]
                self.ubg += [ca.inf for _ in range(len(self.obstacles))]
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            self.lbw += self.state_lower.tolist()
            self.ubw += self.state_upper.tolist()
            self.w0 += [0.0 for _ in range(4)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(4)]
            self.ubg += [0.0 for _ in range(4)]
            
            J += l_k  # accumulate cost
            
        g += [Xk - xr]  # terminal state constraint
        self.lbg += [0.0 for _ in range(4)]
        self.ubg += [0.0 for _ in range(4)]
        
        # Define a function for the constraints
        x_prob, u_prob = self.extract_traj(ca.vertcat(*w))
        self.g_func = ca.Function(
            "g_func",
            [x_prob, u_prob, x0],
            [ca.vertcat(*g)],
            ["X", "U", "x0"],
            ["g"],
        )
        
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False, "verbose": False}
        self.solver = ca.nlpsol("point_mass_mpc", "ipopt", nlp_prob, opts)
        # The projection problem has the same horizon and must be rebuilt if
        # callers redefine the MPC with a different horizon.
        self._feasible_projection_problem = None
        self._linearized_boundary_solvers = {}

    def _reference_decision(self) -> np.ndarray:
        """Return a complete feasible A-to-B trajectory as an IPOPT warm start."""
        z = self.feasible_seed()
        decision = self.traj_to_vec(self.states(z).T, self.controls(z).T)
        return decision.full().reshape(-1)

    def _solve_mpc_solution(self, x0: Sequence[float], warm_start=None):
        initial = np.asarray(self.w0 if warm_start is None else warm_start, dtype=float).reshape(-1)
        solution = self.solver(
            x0=ca.DM(initial),
            lbx=ca.DM(np.asarray(self.lbw, dtype=float).reshape(-1)),
            ubx=ca.DM(np.asarray(self.ubw, dtype=float).reshape(-1)),
            lbg=ca.DM(np.asarray(self.lbg, dtype=float).reshape(-1)),
            ubg=ca.DM(np.asarray(self.ubg, dtype=float).reshape(-1)),
            p=ca.DM(np.asarray(x0, dtype=float).reshape(4)),
        )
        if not self.solver.stats().get("success", False):
            status = self.solver.stats().get("return_status", "unknown")
            raise RuntimeError(f"Point-mass MPC failed ({status}).")
        return solution

    def solve_MPC(self, x0, ret_seq=False):
        """Solve the fixed-goal obstacle MPC from ``x0``."""
        solution = self._solve_mpc_solution(x0, self._reference_decision())
        _, controls = self.extract_traj(solution["x"])
        values = controls.full().T
        return values.reshape(-1) if ret_seq else values[0].copy()

    def solve_MPC_trajectory(self, x0):
        """Return the optimal state and control trajectories from ``x0``."""
        solution = self._solve_mpc_solution(x0, self._reference_decision())
        states, controls = self.extract_traj(solution["x"])
        return states.full().T, controls.full().T

    def _rollout(self, controls: np.ndarray) -> np.ndarray:
        """Return the state trajectory resulting from a sequence of controls."""
        controls = np.asarray(controls, dtype=float).reshape(self.horizon, 2)
        states = np.zeros((self.horizon + 1, 4), dtype=float)
        states[0, :2] = self.start
        for k, acceleration in enumerate(controls):
            states[k + 1] = self.sim_step(states[k], acceleration).full().reshape(4)
        return states

    def _build_reduced_parameterization(self) -> None:
        """Construct a linear map from reduced coordinates to full control and state trajectories."""
        n_controls = 2 * self.horizon
        zero_controls = np.zeros(n_controls)
        free_terminal = self._rollout(zero_controls)[-1]
        terminal_map = np.empty((4, n_controls))
        for column in range(n_controls):
            basis_control = np.zeros(n_controls)
            basis_control[column] = 1.0
            terminal_map[:, column] = self._rollout(basis_control)[-1] - free_terminal

        target_terminal = np.r_[self.goal, np.zeros(2)]
        terminal_delta = target_terminal - free_terminal
        gram = terminal_map @ terminal_map.T
        self.particular_controls = terminal_map.T @ np.linalg.solve(gram, terminal_delta)
        _, singular_values, vh = np.linalg.svd(terminal_map, full_matrices=True)
        tolerance = np.max(terminal_map.shape) * singular_values[0] * np.finfo(float).eps
        rank = int(np.count_nonzero(singular_values > tolerance))
        self.null_basis = vh[rank:].T.copy()

        self.particular_states = self._rollout(self.particular_controls)
        self.state_basis = np.empty((self.horizon + 1, 4, self.reduced_dimension))
        for column in range(self.reduced_dimension):
            controls = self.particular_controls + self.null_basis[:, column]
            self.state_basis[:, :, column] = self._rollout(controls) - self.particular_states

    def controls(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(self.reduced_dimension)
        return (self.particular_controls + self.null_basis @ z).reshape(self.horizon, 2)

    def states(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(self.reduced_dimension)
        return self.particular_states + np.einsum("ksj,j->ks", self.state_basis, z)

    def _build_collision_maps(self) -> None:
        offsets: List[np.ndarray] = []
        maps: List[np.ndarray] = []
        for k in range(self.horizon):
            for substep in range(self.collision_substeps):
                fraction = substep / self.collision_substeps
                tau = fraction * self.dt
                u_offset = self.particular_controls.reshape(self.horizon, 2)[k]
                u_map = self.null_basis.reshape(self.horizon, 2, self.reduced_dimension)[k]
                offsets.append(
                    self.particular_states[k, :2]
                    + tau * self.particular_states[k, 2:]
                    + 0.5 * tau**2 * u_offset
                )
                maps.append(
                    self.state_basis[k, :2, :]
                    + tau * self.state_basis[k, 2:, :]
                    + 0.5 * tau**2 * u_map
                )
        offsets.append(self.particular_states[-1, :2])
        maps.append(self.state_basis[-1, :2, :])
        self.collision_position_offset = np.asarray(offsets)
        self.collision_position_map = np.asarray(maps)

    def collision_positions(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(self.reduced_dimension)
        return self.collision_position_offset + np.einsum(
            "psj,j->ps", self.collision_position_map, z
        )

    @staticmethod
    def _append_box_constraints(
        rows: List[np.ndarray],
        rhs: List[np.ndarray],
        offset: np.ndarray,
        mapping: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        flat_offset = np.asarray(offset).reshape(-1)
        flat_map = np.asarray(mapping).reshape(flat_offset.size, -1)
        rows.extend((flat_map, -flat_map))
        rhs.extend((np.broadcast_to(upper, np.asarray(offset).shape).reshape(-1) - flat_offset,
                    flat_offset - np.broadcast_to(lower, np.asarray(offset).shape).reshape(-1)))

    def _build_linear_constraints(self) -> None:
        rows: List[np.ndarray] = []
        rhs: List[np.ndarray] = []
        control_offset = self.particular_controls.reshape(self.horizon, 2)
        control_map = self.null_basis.reshape(self.horizon, 2, self.reduced_dimension)
        self._append_box_constraints(
            rows, rhs, control_offset, control_map,
            np.array([-self.acceleration_bound] * 2),
            np.array([self.acceleration_bound] * 2),
        )
        self._append_box_constraints(
            rows, rhs, self.particular_states[:, 2:], self.state_basis[:, 2:, :],
            np.array([-self.velocity_bound] * 2),
            np.array([self.velocity_bound] * 2),
        )
        xmin, xmax, ymin, ymax = self.workspace
        self._append_box_constraints(
            rows, rhs, self.collision_position_offset, self.collision_position_map,
            np.array([xmin + self.body_radius, ymin + self.body_radius]),
            np.array([xmax - self.body_radius, ymax - self.body_radius]),
        )
        self.linear_A = np.vstack(rows)
        self.linear_b = np.concatenate(rhs)

    def _build_coordinate_bound_solver(self) -> ca.Function:
        """Create the qpoases LP used for every reduced-coordinate bound."""
        n = self.reduced_dimension
        m = self.linear_A.shape[0]
        structure = {
            "h": ca.Sparsity(n, n),
            "a": ca.Sparsity.dense(m, n),
        }
        return ca.conic(
            "point_mass_coordinate_bound_lp",
            "qpoases",
            structure,
            {"printLevel": "none", "print_time": False},
        )

    def _solve_coordinate_lp(self, objective: np.ndarray) -> np.ndarray:
        """Solve ``min objective.T @ z`` subject to the linear constraints."""
        n = self.reduced_dimension
        m = self.linear_A.shape[0]
        try:
            result = self._coordinate_bound_solver(
                g=ca.DM(objective),
                a=ca.DM(self.linear_A),
                lba=-ca.inf * ca.DM.ones(m, 1),
                uba=ca.DM(self.linear_b),
                lbx=-ca.inf * ca.DM.ones(n, 1),
                ubx=ca.inf * ca.DM.ones(n, 1),
            )
        except RuntimeError as exc:
            raise RuntimeError("qpoases failed while computing coordinate bounds.") from exc
        if not self._coordinate_bound_solver.stats().get("success", False):
            status = self._coordinate_bound_solver.stats().get("return_status", "unknown")
            raise RuntimeError(f"qpoases coordinate-bound LP failed ({status}).")
        return np.asarray(result["x"].full()).reshape(-1)

    def _find_coordinate_bounds(self) -> np.ndarray:
        bounds = np.empty((self.reduced_dimension, 2))
        for index in range(self.reduced_dimension):
            objective = np.zeros(self.reduced_dimension)
            objective[index] = 1.0
            lower_solution = self._solve_coordinate_lp(objective)
            upper_solution = self._solve_coordinate_lp(-objective)
            bounds[index] = (lower_solution[index], upper_solution[index])
        return bounds

    def _casadi_expressions(self, z: ca.SX) -> Dict[str, ca.SX]:
        """Build symbolic trajectories and constraint margins for an NLP."""
        controls = ca.DM(self.particular_controls) + ca.DM(self.null_basis) @ z
        state_offset = ca.DM(self.particular_states.reshape(-1))
        state_map = ca.DM(self.state_basis.reshape(-1, self.reduced_dimension))
        states = state_offset + state_map @ z

        position_offset = ca.DM(self.collision_position_offset.reshape(-1))
        position_map = ca.DM(
            self.collision_position_map.reshape(-1, self.reduced_dimension)
        )
        positions = position_offset + position_map @ z
        positions = ca.reshape(positions, 2, self.collision_position_offset.shape[0])

        obstacle_margins = []
        for obstacle in self.obstacles:
            center = ca.repmat(ca.DM(obstacle.center), 1, positions.size2())
            delta = positions - center
            clearance = obstacle.radius + self.body_radius
            obstacle_margins.append(ca.sum1(delta**2).T - clearance**2)

        linear_margins = ca.DM(self.linear_b) - ca.DM(self.linear_A) @ z
        return {
            "controls": controls,
            "states": states,
            "linear_margins": linear_margins,
            "obstacle_margins": ca.vertcat(*obstacle_margins),
        }

    def linear_margins(self, z: np.ndarray) -> np.ndarray:
        return self.linear_b - self.linear_A @ np.asarray(z, dtype=float)

    def obstacle_margins(self, z: np.ndarray) -> np.ndarray:
        positions = self.collision_positions(z)
        margins = []
        for obstacle in self.obstacles:
            delta = positions - np.asarray(obstacle.center)
            clearance = obstacle.radius + self.body_radius
            margins.append(np.einsum("ij,ij->i", delta, delta) - clearance**2)
        return np.concatenate(margins)

    def obstacle_margin_jacobian(self, z: np.ndarray) -> np.ndarray:
        """Jacobian of the flattened obstacle margins with respect to ``z``."""
        positions = self.collision_positions(z)
        jacobians = []
        for obstacle in self.obstacles:
            delta = positions - np.asarray(obstacle.center)
            jacobians.append(2.0 * np.einsum("ps,psj->pj", delta, self.collision_position_map))
        return np.vstack(jacobians)

    def margins(self, z: np.ndarray) -> np.ndarray:
        return np.r_[self.linear_margins(z), self.obstacle_margins(z)]

    def is_feasible(self, x0, Nsim=None, tol=1e-3, max_steps=50) -> bool:
        """Check MPC feasibility of an initial state.

        Reduced trajectory coordinates are still accepted internally for
        backward compatibility with the trajectory-generation helpers.
        """
        values = np.asarray(x0, dtype=float).reshape(-1)
        if values.size == self.reduced_dimension:
            return bool(np.min(self.margins(values)) >= -tol)
        if values.size != 4:
            raise ValueError("x0 must have four state components.")
        try:
            print("Checking feasibility of initial state:", values)
            self._solve_mpc_solution(values, self._reference_decision())
        except (RuntimeError, ValueError):
            return False
        return True

    def _waypoint_guess(self) -> np.ndarray:
        waypoints = np.array([
            self.start,
            (-3.2, -2.30),
            (-1.6, 1.15),
            (0.0, -1.15),
            (1.6, 2.30),
            (3.2, 0.0),
            self.goal,
        ])
        waypoint_t = np.linspace(0.0, self.horizon * self.dt, len(waypoints))
        target_positions = self._clamped_cubic_interpolation(
            waypoint_t,
            waypoints,
            np.arange(self.horizon + 1) * self.dt,
        )
        position_map = self.state_basis[:, :2, :].reshape(-1, self.reduced_dimension)
        return np.linalg.lstsq(
            position_map,
            (target_positions - self.particular_states[:, :2]).reshape(-1),
            rcond=None,
        )[0]

    @staticmethod
    def _clamped_cubic_interpolation(
        knot_times: np.ndarray,
        knot_values: np.ndarray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a cubic spline with zero derivative at both endpoints."""
        knot_times = np.asarray(knot_times, dtype=float)
        knot_values = np.asarray(knot_values, dtype=float)
        intervals = np.diff(knot_times)
        num_knots = knot_times.size

        system = np.zeros((num_knots, num_knots))
        rhs = np.zeros_like(knot_values)
        slopes = np.diff(knot_values, axis=0) / intervals[:, None]
        system[0, :2] = (2.0 * intervals[0], intervals[0])
        rhs[0] = 6.0 * slopes[0]
        system[-1, -2:] = (intervals[-1], 2.0 * intervals[-1])
        rhs[-1] = -6.0 * slopes[-1]
        for index in range(1, num_knots - 1):
            previous = intervals[index - 1]
            following = intervals[index]
            system[index, index - 1:index + 2] = (
                previous,
                2.0 * (previous + following),
                following,
            )
            rhs[index] = 6.0 * (slopes[index] - slopes[index - 1])
        second_derivatives = np.linalg.solve(system, rhs)

        output = np.empty((len(query_times), knot_values.shape[1]))
        for output_index, time in enumerate(query_times):
            segment = min(np.searchsorted(knot_times, time, side="right") - 1, num_knots - 2)
            segment = max(segment, 0)
            step = intervals[segment]
            left_weight = (knot_times[segment + 1] - time) / step
            right_weight = (time - knot_times[segment]) / step
            output[output_index] = (
                left_weight * knot_values[segment]
                + right_weight * knot_values[segment + 1]
                + ((left_weight**3 - left_weight) * second_derivatives[segment]
                   + (right_weight**3 - right_weight) * second_derivatives[segment + 1])
                * step**2
                / 6.0
            )
        return output

    def feasible_seed(self) -> np.ndarray:
        """Find and cache a smooth feasible trajectory with a CasADi/IPOPT NLP."""
        if self._feasible_seed is not None:
            return self._feasible_seed.copy()
        guess = self._waypoint_guess()
        target_positions = self.states(guess)[:, :2]

        z = ca.SX.sym("seed_z", self.reduced_dimension)
        expressions = self._casadi_expressions(z)
        state_positions = ca.reshape(
            expressions["states"], 4, self.horizon + 1
        )[:2, :]
        tracking_error = state_positions - ca.DM(target_positions.T)
        objective = (
            ca.sumsqr(tracking_error) / target_positions.size
            + 2e-3 * ca.sumsqr(expressions["controls"]) / (2 * self.horizon)
        )
        constraints = ca.vertcat(
            expressions["linear_margins"],
            expressions["obstacle_margins"],
        )
        problem = {"x": z, "f": objective, "g": constraints}
        options = {
            "print_time": False,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 2500,
                "tol": 1e-9,
            },
        }
        solver = ca.nlpsol("point_mass_feasible_seed", "ipopt", problem, options)
        num_linear = self.linear_b.size
        num_obstacle = self.obstacle_margins(guess).size
        num_constraints = num_linear + num_obstacle
        try:
            solution = solver(
                x0=ca.DM(guess),
                lbg=ca.DM(np.r_[np.zeros(num_linear), 0.05 * np.ones(num_obstacle)]),
                ubg=ca.inf * ca.DM.ones(num_constraints, 1),
            )
        except RuntimeError as exc:
            status = solver.stats().get("return_status", "unknown")
            raise RuntimeError(f"IPOPT feasible-seed NLP failed ({status}).") from exc
        candidate = np.asarray(solution["x"].full()).reshape(-1)
        if not solver.stats().get("success", False) or not self.is_feasible(
            candidate, tol=2e-6
        ):
            status = solver.stats().get("return_status", "unknown")
            raise RuntimeError(
                "Could not construct the initial feasible path; try increasing the horizon. "
                f"IPOPT status: {status}"
            )
        self._feasible_seed = candidate
        return self._feasible_seed.copy()

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
        # Generate a random array of length ``n`` with optional spiking of some entries.
        values = (
            np.random.exponential(base_scale, size=n)
            if positive
            else np.random.normal(0.0, base_scale, size=n)
        )
        if n == 0 or np.random.random() > spike_probability:
            return values, np.empty(0, dtype=int)
        count = np.random.randint(min_spikes, min(max_spikes, n) + 1)
        indices = np.random.choice(n, size=count, replace=False)
        values[indices] *= np.random.uniform(*spike_factor_range, size=count)
        return values, indices

    def seek_x0(
        self,
        N,
        close_to_bounds=True,
        u0_active_bound=False,
        plot_results=False,
        return_u_seq=False,
        slack_penalty="l1",
    ):
        """Find a feasible initial state whose trajectory reaches B."""
        slack_penalty = str(slack_penalty).lower()
        if slack_penalty not in {"l1", "quadratic"}:
            raise ValueError("slack_penalty must be either 'l1' or 'quadratic'.")

        def weighted_slack_cost(slacks, weights):
            regularization = 1e-6 * ca.dot(slacks, slacks)
            if slack_penalty == "l1":
                return ca.dot(ca.DM(weights), slacks) + regularization
            return slacks.T @ np.diag(weights) @ slacks + regularization
        
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 4)  # state variable
        u = ca.SX.sym('u', 2)  # control variable
        x0 = ca.SX.sym('x0', 4)  # initial state parameter
        
        X = ca.SX.sym('X', 4, N+1)  # state trajectory
        U = ca.SX.sym('U', 2, N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        
        x_next = self.mpc_step(x, u)
        xr = ca.DM(self.goal.tolist() + [0.0, 0.0])  # reference state (goal with zero velocity)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        lbw = []    # lower bounds on decision variables
        ubw = []    # upper bounds on decision variables
        w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        lbg = []    # lower bounds on constraints
        ubg = []    # upper bounds on constraints
        
        
        x_max = ca.DM(self.state_upper.tolist())
        x_min = ca.DM(self.state_lower.tolist())
        u_max = ca.DM([self.acceleration_bound, self.acceleration_bound])
        u_min = ca.DM([-self.acceleration_bound, -self.acceleration_bound])
        
        
        Xk = ca.SX.sym('X0', 4)  # initial state variable
        w += [Xk]
        lbw += [-ca.inf for _ in range(4)]  # no lower bound on initial state
        ubw += [ca.inf for _ in range(4)]  # no upper bound on initial state
        w0 += [0.0 for _ in range(4)]
        
        s_x0 = ca.SX.sym('slack_x0', 4, 2)
        s_x0_var = ca.vec(s_x0)
        w += [s_x0_var]
        lbw += [0.0 for _ in range(8)]
        ubw += [ca.inf for _ in range(8)]
        w0 += [0.0 for _ in range(8)]
        
        if close_to_bounds:
            g += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
            lbg += [0.0 for _ in range(8)]
            ubg += [0.0 for _ in range(8)]
        else:
            g += [Xk - x_max + s_x0[:, 0], -Xk + x_min + s_x0[:, 1]]
            lbg += [-ca.inf for _ in range(8)]
            ubg += [0.0 for _ in range(8)]
            
        X_syms = [Xk]
        U_syms = []
        x_slacks = [s_x0_var]
        u_slacks = []
        g_slacks = []
        # Each residual block stores upper residuals first, then lower residuals.
        # Keep the block sizes so random spiking can choose at most one side per bound.
        x_slack_block_sizes = [4]
        u_slack_block_sizes = [2]
        
        for k in range(N):
            uk = ca.SX.sym('u_' + str(k), 2)
            w += [uk]
            lbw += [-ca.inf] * 2  # control limits
            ubw += [ca.inf] * 2
            w0 += [0.0 for _ in range(2)]     # initial guess
            
            s_uk = ca.SX.sym('slack_u_' + str(k), 2, 2)
            s_uk_var = ca.vec(s_uk)
            w += [s_uk_var]
            lbw += [0.0, 0.0, 0.0, 0.0]  # slack variables for control limits
            ubw += [ca.inf, ca.inf, ca.inf, ca.inf]
            w0 += [0.0, 0.0, 0.0, 0.0]

            g += [uk - u_max + s_uk[:, 0], -uk + u_min + s_uk[:, 1]]
            lbg += [0.0, 0.0, 0.0, 0.0]
            ubg += [0.0, 0.0, 0.0, 0.0]
            
            Xk_next, l_k = F(Xk, uk)
            
            # Add obstacle avoidance constraints at each substep
            for substep in range(self.collision_substeps):
                tau = substep * self.dt_mpc / self.collision_substeps
                position = Xk[:2] + tau * Xk[2:] + 0.5 * tau**2 * uk
                obstacle_margin = self._symbolic_obstacle_margins(position)
                
                # # Add slack variables for obstacle avoidance
                # s_obstacle = ca.SX.sym('slack_obstacle_' + str(k) + '_' + str(substep), len(self.obstacles))
                # s_obstacle_var = ca.vec(s_obstacle)
                # w += [s_obstacle_var]
                # lbw += [0.0 for _ in range(2 * len(self.obstacles))]  # slack variables for obstacle avoidance
                # ubw += [ca.inf for _ in range(2 * len(self.obstacles))]
                
                g += [obstacle_margin]  # obstacle avoidance constraint
                lbg += [0.0 for _ in range(len(self.obstacles))]
                ubg += [ca.inf for _ in range(len(self.obstacles))]
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            lbw += [-ca.inf for _ in range(4)]
            ubw += [ca.inf for _ in range(4)]
            w0 += [0.0 for _ in range(4)]
            
            s_xk = ca.SX.sym('slack_x_' + str(k), 4, 2)
            s_xk_var = ca.vec(s_xk)
            w += [s_xk_var]
            lbw += [0.0 for _ in range(8)]
            ubw += [ca.inf for _ in range(8)]
            w0 += [0.0 for _ in range(8)]

            if close_to_bounds:
                g += [Xk - x_max + s_xk[:, 0], -Xk + x_min + s_xk[:, 1]]
                lbg += [0.0 for _ in range(8)]
                ubg += [0.0 for _ in range(8)]
            else:
                g += [Xk - x_max + s_xk[:, 0], -Xk + x_min + s_xk[:, 1]]
                lbg += [-ca.inf for _ in range(8)]
                ubg += [0.0 for _ in range(8)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            lbg += [0.0 for _ in range(4)]
            ubg += [0.0 for _ in range(4)]
            
            J += l_k  # accumulate cost
            
            X_syms.append(Xk)
            U_syms.append(uk)
            x_slacks.append(s_xk_var)
            u_slacks.append(s_uk_var)
            # g_slacks.append(s_obstacle_var)
            x_slack_block_sizes.append(4)
            u_slack_block_sizes.append(2)
        
        if close_to_bounds:
            x_slacks_vec = ca.vertcat(*x_slacks)
            rho, _ = self.spiky_residual_weights(
                x_slack_block_sizes,
                min_spikes=1,
                max_spikes=1,
                base_scale=1.0,
                spike_factor_range=(100, 1000),
                spike_probability=1.0,
            )
            J += weighted_slack_cost(x_slacks_vec, rho)
        else:
            J += 1e-6 * ca.dot(ca.vertcat(*x_slacks), ca.vertcat(*x_slacks))
        
        if u0_active_bound:
            # Push the first control input to the upper/lower bound with a spiky penalty.
            side = np.random.randint(2)
            J += 1e6 * ca.dot(u_slacks[0][side], u_slacks[0][side]) + 1e-6 * ca.dot(ca.vertcat(*u_slacks), ca.vertcat(*u_slacks))
        else: 
            u_slacks_vec = ca.vertcat(*u_slacks)
            rho, _ = self.spiky_residual_weights(
                u_slack_block_sizes,
                min_spikes=1,
                max_spikes=1,
                base_scale=1.0,
                spike_factor_range=(100, 1000),
                spike_probability=1.0,
            )
        
        # g_slacks_vec = ca.vertcat(*g_slacks)
        # rho, _ = self.spiky_array(
        #     g_slacks_vec.shape[0],
        #     min_spikes=1,
        #     max_spikes=1,
        #     base_scale=1.0,
        #     spike_factor_range=(100, 1000),
        #     spike_probability=1.0,
        # )
        # J += weighted_slack_cost(g_slacks_vec, rho)
        
        X_stack = ca.hcat(X_syms)
        U_stack = ca.hcat(U_syms)
        extract = ca.Function("extract_extreme_traj", [ca.vertcat(*w)], [X_stack, U_stack])
        slack_stack = ca.vertcat(*(x_slacks + u_slacks + g_slacks))
        extract_slacks = ca.Function("extract_slacks", [ca.vertcat(*w)], [slack_stack])

        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        opts = {
            "expand": False,
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
        try:
            solution = solver(
                x0=ca.DM(w0),
                lbx=ca.DM(np.asarray(lbw, dtype=float).reshape(-1)),
                ubx=ca.DM(np.asarray(ubw, dtype=float).reshape(-1)),
                lbg=ca.DM(np.asarray(lbg, dtype=float).reshape(-1)),
                ubg=ca.DM(np.asarray(ubg, dtype=float).reshape(-1)),
                # p=ca.DM(weights),
            )
        except RuntimeError:
            empty = (np.full(4, np.nan), np.full(2, np.nan), "Solve_Failed")
            return (*empty[:2], np.full(2 * self.N, np.nan), empty[2]) if return_u_seq else empty
        X_opt, U_opt = extract(solution["x"])
        status = solver.stats().get("return_status", "unknown")
        x0_opt = X_opt[:, 0].full().reshape(4)
        controls = U_opt.full().T
        if return_u_seq:
            return x0_opt, controls[0].copy(), controls.reshape(-1), status
        return x0_opt, controls[0].copy(), status

    def collect_feasible_samples(
        self,
        x_max=None,
        close_to_bounds=True,
        u0_active_bound=False,
        num_samples=100,
        max_attempts=1000,
        plot_results=True,
        slack_penalty="l1",
    ):
        """Collect feasible four-dimensional initial states, as CartPole does."""
        samples = []
        controls = []
        attempts = 0
        successes = 0
        for attempts in range(1, max_attempts + 1):
            if len(samples) >= num_samples:
                break
            x0, u0, status = self.seek_x0(
                self.N,
                close_to_bounds=close_to_bounds,
                u0_active_bound=u0_active_bound,
                plot_results=False,
                slack_penalty=slack_penalty,
            )
            if status not in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}:
                print(f"Seek_x0 failed on attempt {attempts} with status: {status}")
                continue
            successes += 1
            print(f"Feasible samples found: {len(samples) + 1}/{num_samples}, attempts: {attempts}, successes: {successes}")
            if not samples or min(np.linalg.norm(x0 - old) for old in samples) > 1e-6:
                samples.append(x0)
                controls.append(u0)
                if len(samples) >= num_samples:
                    break
        result = np.asarray(samples, dtype=float).reshape(-1, 4)
        self.last_feasible_sample_controls = np.asarray(controls, dtype=float).reshape(-1, 2)
        self.last_feasible_sampling_stats = {
            "slack_penalty": str(slack_penalty).lower(),
            "requested_samples": int(num_samples),
            "returned_samples": int(result.shape[0]),
            "attempts": int(attempts),
            "successful_extreme_solves": int(successes),
            "feasible_candidates": int(result.shape[0]),
        }
        if plot_results:
            self.plot_feasible_set(samples=result, show=True)
        return result

    @staticmethod
    def _condense_dynamics_and_constraints(A_seq, B_seq, G_x_seq, G_u_seq):
        """Condense ``dx[k+1] = A[k]dx[k] + B[k]du[k]``.
        Omega = [A[0] @ A[1] @ ... @ A[N-1]]
        Gamma = [A[0] @ A[1] @ ... @ A[N-1] @ B[0], A[1] @ ... @ A[N-1] @ B[1], ..., B[N-1]]
        Condense ``g[k] = G_x[k]dx[k] + G_u[k]du[k]``.
        G = [G_x[0], G_x[1], ..., G_x[N-1]]
        H = [G_u[0], G_u[1], ..., G_u[N-1]]
        Explicit the x[k] evolution in the constraints: g = G @ dx + H @ du = G @ (Omega @ dx[0] + Gamma @ du) + H @ du = (G @ Omega) @ dx[0] + (G @ Gamma + H) @ du
        """
        N = A_seq.shape[0]
        state_dim = A_seq.shape[1]
        control_dim = B_seq.shape[2]

        Omega = np.eye(state_dim)
        Gamma = np.zeros((state_dim, N * control_dim))
        for k in range(N):
            Omega = A_seq[k] @ Omega
            product = np.eye(state_dim)
            for j in range(k + 1, N):
                product = A_seq[j] @ product
            Gamma[:, k * control_dim:(k + 1) * control_dim] = product @ B_seq[k]

        constraint_dim = G_x_seq.shape[1]
        control_dim = G_u_seq.shape[2]

        Gx = np.zeros((N * constraint_dim, N * control_dim))
        Gu = np.zeros((N * constraint_dim, N * control_dim))
        for k in range(N):
            Gx[k * constraint_dim:(k + 1) * constraint_dim, k * control_dim:(k + 1) * control_dim] = G_x_seq[k]
            Gu[k * constraint_dim:(k + 1) * constraint_dim, k * control_dim:(k + 1) * control_dim] = G_u_seq[k]

        # Explicit the x[k] evolution in the constraints: 
        # g = Gx @ dx + Gu @ du = Gx @ (Omega @ dx[0] + Gamma @ du) + Gu @ du = (Gx @ Omega) @ dx[0] + (Gx @ Gamma + Gu) @ du
        G_condensed = Gx @ Omega
        H_condensed = Gx @ Gamma + Gu
        
        return G_condensed, H_condensed

    def _linearized_ocp_at_solution(self, solution):
        """Return the local linearized OCP model around a nominal solution.

        The point-mass dynamics are already linear and time invariant, so the
        only nonconvex part that needs linearization is obstacle avoidance.
        ``Omega`` and ``Gamma`` condense the state trajectory, while ``F``,
        ``G``, and ``v`` define the affine obstacle-margin approximation

        ``margin(x0, u) ~= F @ x0 + G @ u + v``.
        """
        states_dm, controls_dm = self.extract_traj(solution["x"])
        X_nominal = np.asarray(states_dm.full(), dtype=float)
        U_nominal = np.asarray(controls_dm.full(), dtype=float)

        # Linearize the dynamics around the nominal trajectory.
        x_sym = ca.SX.sym("linearization_x", 4)
        u_sym = ca.SX.sym("linearization_u", 2)
        next_state = self.mpc_step(x_sym, u_sym)
        dynamics_jacobian = ca.Function(
            "point_mass_dynamics_jacobian",
            [x_sym, u_sym],
            [ca.jacobian(next_state, x_sym), ca.jacobian(next_state, u_sym)],
        )
        A_dm, B_dm = dynamics_jacobian(X_nominal[:, 0], U_nominal[:, 0])
        A = np.asarray(A_dm.full(), dtype=float)
        B = np.asarray(B_dm.full(), dtype=float)

        # Condense the dynamics over the horizon.
        n_u = 2 * self.N
        omega = np.zeros((4 * self.N, 4), dtype=float)
        gamma = np.zeros((4 * self.N, n_u), dtype=float)
        state_map = np.eye(4)
        control_map = np.zeros((4, n_u), dtype=float)
        for k in range(self.N):
            state_map = A @ state_map
            control_map = A @ control_map
            control_map[:, 2 * k:2 * k + 2] += B
            block = slice(4 * k, 4 * (k + 1))
            omega[block] = state_map
            gamma[block] = control_map

        constraint_state_maps = []
        constraint_control_maps = []
        constraint_offsets = []
        obstacle_centers = np.asarray(
            [obstacle.center for obstacle in self.obstacles], dtype=float
        ).reshape(-1, 2)
        clearances = np.asarray(
            [obstacle.radius + self.body_radius for obstacle in self.obstacles],
            dtype=float,
        )
        for k in range(self.N + 1):
            if k == 0:
                state_x_map = np.eye(4)
                state_u_map = np.zeros((4, n_u), dtype=float)
            else:
                block = slice(4 * (k - 1), 4 * k)
                state_x_map = omega[block]
                state_u_map = gamma[block]

            substeps = range(self.collision_substeps) if k < self.N else (0,)
            for substep in substeps:
                tau = substep * self.dt_mpc / self.collision_substeps
                position_selector = np.array(
                    [[1.0, 0.0, tau, 0.0], [0.0, 1.0, 0.0, tau]]
                )
                position_x_map = position_selector @ state_x_map
                position_u_map = position_selector @ state_u_map
                nominal_position = position_selector @ X_nominal[:, k]
                if k < self.N:
                    acceleration_map = np.zeros((2, n_u), dtype=float)
                    acceleration_map[:, 2 * k:2 * k + 2] = np.eye(2)
                    position_u_map += 0.5 * tau**2 * acceleration_map
                    nominal_position += 0.5 * tau**2 * U_nominal[:, k]

                displacement = nominal_position - obstacle_centers
                gradients = 2.0 * displacement
                margins = (
                    np.einsum("ij,ij->i", displacement, displacement)
                    - clearances**2
                )
                constraint_state_maps.append(gradients @ position_x_map)
                constraint_control_maps.append(gradients @ position_u_map)
                constraint_offsets.append(margins - gradients @ nominal_position)

        F = np.vstack(constraint_state_maps)
        G = np.vstack(constraint_control_maps)
        v = np.concatenate(constraint_offsets)
        return {
            "Omega": omega,
            "Gamma": gamma,
            "F": F,
            "G": G,
            "v": v,
            "X_nominal": X_nominal,
            "U_nominal": U_nominal,
        }

    def _get_linearized_boundary_solver(self, n_variables, n_constraints, plugin):
        key = (int(n_variables), int(n_constraints))
        solver = self._linearized_boundary_solvers.get(key)
        if solver is None:
            structure = {
                "h": ca.Sparsity(n_variables, n_variables),
                "a": ca.Sparsity.dense(n_constraints, n_variables),
            }
            solver = ca.conic(
                f"point_mass_hr_boundary_{n_variables}_{n_constraints}",
                "qpoases",
                structure,
                {"printLevel": "none", "print_time": False},
            )
            self._linearized_boundary_solvers[key] = solver
        return solver

    def _linearized_boundary_step(
        self,
        x,
        direction,
        linear_ocp,
        initial_state_bounds,
        prediction_state_bounds=None,
        lp_solver="qpoases",
    ):
        """Find the local feasible-set boundary along ``direction``.

        The LP variables are ``[delta_u, alpha]``, where ``delta_u`` is a
        deviation from the nominal MPC controls and the candidate initial state
        is ``x + alpha * direction``.  At every collision-check point, impose

        ``h(p_nom) + grad_h(p_nom) @ (p - p_nom) >= 0``

        for each circular-obstacle margin ``h(p)``.  The position ``p`` is
        affine in both ``delta_u`` and ``alpha`` through the condensed dynamics.
        """
        x = np.asarray(x, dtype=float).reshape(4)
        direction = np.asarray(direction, dtype=float).reshape(4)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(direction)):
            return None
        if np.linalg.norm(direction) <= 1e-14:
            return None

        initial_lower = self.state_lower.copy()
        initial_upper = self.state_upper.copy()
        if initial_state_bounds is not None:
            requested = np.asarray(initial_state_bounds, dtype=float).reshape(4)
            initial_lower = np.maximum(initial_lower, -requested)
            initial_upper = np.minimum(initial_upper, requested)
        prediction_lower = self.state_lower.copy()
        prediction_upper = self.state_upper.copy()
        if prediction_state_bounds is not None:
            requested = np.asarray(prediction_state_bounds, dtype=float).reshape(4)
            prediction_lower = np.maximum(prediction_lower, -requested)
            prediction_upper = np.minimum(prediction_upper, requested)

        # Extract the condensed dynamics and affine obstacle model.
        omega = np.asarray(linear_ocp["Omega"], dtype=float)
        gamma = np.asarray(linear_ocp["Gamma"], dtype=float)
        F = np.asarray(linear_ocp["F"], dtype=float)
        G = np.asarray(linear_ocp["G"], dtype=float)
        v = np.asarray(linear_ocp["v"], dtype=float).reshape(-1)
        X_nominal = np.asarray(linear_ocp["X_nominal"], dtype=float)
        U_nominal = np.asarray(linear_ocp["U_nominal"], dtype=float)

        n_u = 2 * self.N
        n_collision_points = self.N * self.collision_substeps + 1
        n_obstacle_constraints = n_collision_points * len(self.obstacles)
        if (
            omega.shape != (4 * self.N, 4)
            or gamma.shape != (4 * self.N, n_u)
            or F.shape != (n_obstacle_constraints, 4)
            or G.shape != (n_obstacle_constraints, n_u)
            or v.shape != (n_obstacle_constraints,)
            or X_nominal.shape != (4, self.N + 1)
            or U_nominal.shape != (2, self.N)
        ):
            raise ValueError("Inconsistent linearized OCP dimensions.")

        nominal_controls = U_nominal.T.reshape(-1)
        predicted_base = omega @ x + gamma @ nominal_controls
        rows, lba, uba = [], [], []
        tolerance = 1e-9

        def add_box(base, control_map, direction_map, box_lower, box_upper):
            variation = np.c_[control_map, direction_map]
            rows.extend((variation, -variation))
            uba.extend((box_upper - base + tolerance,
                        base - box_lower + tolerance))
            lba.extend((np.full(4, -np.inf), np.full(4, -np.inf)))

        add_box(
            x,
            np.zeros((4, n_u)),
            direction,
            initial_lower,
            initial_upper,
        )
        for k in range(1, self.N + 1):
            block = slice(4 * (k - 1), 4 * k)
            add_box(
                predicted_base[block],
                gamma[block],
                omega[block] @ direction,
                prediction_lower,
                prediction_upper,
            )

        target = np.r_[self.goal, np.zeros(2)]
        terminal = slice(4 * (self.N - 1), 4 * self.N)
        terminal_row = np.c_[gamma[terminal], omega[terminal] @ direction]
        terminal_rhs = target - predicted_base[terminal]
        # rows.append(terminal_row)
        # lba.append(terminal_rhs)
        # uba.append(terminal_rhs)

        # F @ x0 + G @ u + v is the affine obstacle-margin model.  Substitute
        # x0 = x + alpha*direction and u = U_nominal + delta_u.
        obstacle_base = F @ x + G @ nominal_controls + v
        rows.append(np.c_[G, F @ direction])
        lba.append(-obstacle_base - tolerance)
        uba.append(np.full(n_obstacle_constraints, np.inf))

        A_lp = np.vstack(rows)
        lba_vec = np.concatenate(lba)
        uba_vec = np.concatenate(uba)
        objective = np.zeros(n_u + 1)
        objective[-1] = -1.0
        lbx = np.r_[-self.acceleration_bound - nominal_controls, 0.0]
        ubx = np.r_[self.acceleration_bound - nominal_controls, np.inf]
        solver = self._get_linearized_boundary_solver(
            n_u + 1, A_lp.shape[0], lp_solver
        )
        try:
            result = solver(
                g=objective,
                a=A_lp,
                lba=lba_vec,
                uba=uba_vec,
                lbx=lbx,
                ubx=ubx,
            )
        except RuntimeError:
            return None
        if not solver.stats().get("success", True):
            return None
        alpha = float(result["x"][-1])
        return alpha if np.isfinite(alpha) and alpha >= 0.0 else None

    def collect_linearized_hit_and_run_samples(
        self,
        num_samples=100,
        num_feasible_seeds=5,
        max_seed_attempts=1000,
        state_per_seed=20,
        random_seed=None,
        x_max=None,
        plot_results=True,
        show=True,
        lp_solver="qpoases",
    ):
        """Sample feasible states in an OCP linearized at an initial seed.

        The nonlinear MPC solution and its local affine OCP are computed only
        at a feasible seed collected at the start. Candidate states advance the
        Hit-and-Run chain, but are never used as new linearization anchors.
        """
        # Collect the only states that may serve as linearization anchors.
        # Include the seeds to the feasible states
        rng = np.random.default_rng(random_seed)
        seeds, _, seed_stats = self.sample_uniform(
            num_samples=num_feasible_seeds,
            max_attempts=max_seed_attempts,
            rng=rng,
        )
        
        # For each feasible seed, generate ns samples along the linearized OCP chord.
        # If no feasible seeds are found, return an empty array.        
        if seeds.shape[0] == 0:
            return np.empty((0, 4))

        accepted = []
        # raw_candidates = []
        # raw_feasible = []
        # Select an original feasible seed for which the full nonlinear MPC can
        # be solved. This seed remains the fixed linearization anchor.
        for seed in list(seeds):
            seed_solution = self._solve_mpc_solution(seed, self._reference_decision())
            # Linearize the OCP at the feasible seed solution. 
            linear_ocp = self._linearized_ocp_at_solution(seed_solution)
            accepted += [seed.copy()]
            raw_candidates = []
            raw_feasible = []
            feas_state = 0
            boundary_failures = 0
            attempts = 0
            

            # Advance the chain within the fixed seed-based linearized OCP until it
            # has completed enough successful updates for burn-in and collection.
            # The attempt limit bounds retries caused by boundary or projection
            # failures.
            while feas_state < state_per_seed and attempts < max_seed_attempts:
                attempts += 1
                print(f"Attempt {attempts}/{max_seed_attempts}")
                
                # The boundary LP is always built from `linear_ocp`, which was
                # created from `seed_solution`; `current` only selects the point on
                # the chord and is never passed to the nonlinear linearization step.
                # Draw a unit direction in the four-dimensional state space.
                direction = rng.normal(size=4)
                direction /= np.linalg.norm(direction)

                # Find both endpoints of the chord through the current chain state.
                # Skip the iteration if either boundary LP fails.
                positive = self._linearized_boundary_step(
                    seed, direction, linear_ocp, x_max, lp_solver=lp_solver
                )
                negative = self._linearized_boundary_step(
                    seed, -direction, linear_ocp, x_max, lp_solver=lp_solver
                )
                if positive is None or negative is None:
                    boundary_failures += 1
                    continue
                print(f"Boundary LPs succeeded: +{positive:.4f}, -{negative:.4f}")
                # After solving the LPs, sample target_updates states along the chord between the two endpoints.  
                # The LPs guarantee that the endpoints are feasible for the linearized OCP, 
                # but not necessarily for the original nonlinear MPC. 
                
                candidate = seed + rng.uniform(-negative, positive) * direction
                raw_candidates.append(candidate.copy())
                # Solve the projection nonlinear MPC to verify feasibility. 
                # If the state is already feasible, the projection is a no-op and returns the same state.
                # If the state is infeasible, the projection attempts to find a nearby feasible state.
                feasible_state = self.project_onto_feasible_set(candidate)
                raw_feasible.append(feasible_state)

                accepted.append(feasible_state)
                feas_state += 1
                
                
                print(f"Feasible samples {feas_state}/{state_per_seed}")
                print(f"Collected {len(accepted)}/{num_samples} samples")
            
        result = np.asarray(accepted, dtype=float).reshape(-1, 4)
        print(f"Collected {result.shape[0]} feasible samples after {attempts} attempts.")
        self.last_linearized_sampling_stats = {
            "requested_samples": int(num_samples),
            "returned_samples": int(result.shape[0]),
            "seeds": seeds,
            "seed_attempts": int(seed_stats["attempts"]),
            "raw_candidates": np.asarray(raw_candidates, dtype=float).reshape(-1, 4),
            "raw_nonlinear_feasible": np.asarray(raw_feasible, dtype=bool),
            "chain_attempts": int(attempts),
            "lp_solver": str(lp_solver),
        }
        if plot_results:
            print("pippo")
            self.plot_linearized_hit_and_run_samples(
                result,
                seeds=seeds,
                show=show,
            )
        return result

    def _build_feasible_projection_problem(self):
        """Build the nearest-feasible-initial-state NLP for IPOPT."""
        # Define optimization variables and parameters
        x = ca.SX.sym('x', 4)  # state variable
        u = ca.SX.sym('u', 2)  # control variable
        x0 = ca.SX.sym('x0', 4)  # initial state parameter
        
        X = ca.SX.sym('X', 4, self.N+1)  # state trajectory
        U = ca.SX.sym('U', 2, self.N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        
        x_next = self.mpc_step(x, u)
        xr = ca.DM(self.goal.tolist() + [0.0, 0.0])  # reference state (goal with zero velocity)
        l = (x - xr).T @ self.Q @ (x - xr) + u.T @ self.R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        lbw = []    # lower bounds on decision variables
        ubw = []    # upper bounds on decision variables
        w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        lbg = []    # lower bounds on constraints
        ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', 4)  # initial state variable
        w += [Xk]
        lbw += self.state_lower.tolist()
        ubw += self.state_upper.tolist()
        w0 += [0.0 for _ in range(4)]
        J += ca.dot(Xk - x0, Xk - x0)  # cost for initial state deviation
        for k in range(self.N):
            uk = ca.SX.sym('u_' + str(k), 2)
            w += [uk]
            lbw += [-self.acceleration_bound] * 2  # control limits
            ubw += [self.acceleration_bound] * 2
            w0 += [0.0 for _ in range(2)]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            # Add obstacle avoidance constraints at each substep
            for substep in range(self.collision_substeps):
                tau = substep * self.dt_mpc / self.collision_substeps
                position = Xk[:2] + tau * Xk[2:] + 0.5 * tau**2 * uk
                obstacle_margin = self._symbolic_obstacle_margins(position)
                g += [obstacle_margin]
                lbg += [0.0 for _ in range(len(self.obstacles))]
                ubg += [ca.inf for _ in range(len(self.obstacles))]
            
            Xk = ca.SX.sym('X_' + str(k+1), 4)
            w += [Xk]
            lbw += self.state_lower.tolist()
            ubw += self.state_upper.tolist()
            w0 += [0.0 for _ in range(4)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            lbg += [0.0 for _ in range(4)]
            ubg += [0.0 for _ in range(4)]
            
            J += l_k  # accumulate cost
            
        g += [Xk - xr]  # terminal state constraint
        lbg += [0.0 for _ in range(4)]
        ubg += [0.0 for _ in range(4)]
            
        problem = {
            "x": ca.vertcat(*w),
            "p": x0,
            "f": J,
            "g": ca.vertcat(*g),
        }
        solver = ca.nlpsol(
            "point_mass_feasible_projection",
            "ipopt",
            problem,
            {
                "print_time": False,
                "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 2000, "tol": 1e-8},
            },
        )
        return {
            "solver": solver,
            "extract": self.extract_traj,
            "lbg": ca.DM(np.asarray(lbg, dtype=float).reshape(-1)),
            "ubg": ca.DM(np.asarray(ubg, dtype=float).reshape(-1)),
        }

    def project_onto_feasible_set(
        self,
        x_target,
        feasible_reference=None,
        reference_solution=None,
    ):
        """Locally project an initial state onto the obstacle-MPC feasible set."""
        if self._projection_problem is None:
            self._projection_problem = self._build_feasible_projection_problem()
        problem = self._projection_problem
        if reference_solution is not None:
            initial = reference_solution["x"]
        elif feasible_reference is not None:
            reference_solution = self._solve_mpc_solution(
                feasible_reference, self._reference_decision()
            )
            initial = reference_solution["x"]
        else:
            initial = self._reference_decision()
        try:
            solution = problem["solver"](
                x0=ca.DM(np.asarray(initial, dtype=float).reshape(-1)),
                lbx=ca.DM(np.asarray(self.lbw, dtype=float).reshape(-1)),
                ubx=ca.DM(np.asarray(self.ubw, dtype=float).reshape(-1)),
                lbg=ca.DM(np.asarray(problem["lbg"], dtype=float).reshape(-1)),
                ubg=ca.DM(np.asarray(problem["ubg"], dtype=float).reshape(-1)),
                p=ca.DM(np.asarray(x_target, dtype=float).reshape(4)),
            )
        except RuntimeError as exc:
            raise RuntimeError("IPOPT feasible-set projection failed.") from exc
        if not problem["solver"].stats().get("success", False):
            status = problem["solver"].stats().get("return_status", "unknown")
            raise RuntimeError(f"IPOPT feasible-set projection failed ({status}).")
        X, _ = problem["extract"](solution["x"])
        return X[:, 0].full().reshape(4)

    @staticmethod
    def _normalize_state_batch(x0_batch):
        values = np.asarray(x0_batch, dtype=float)
        if values.size == 0:
            return np.empty((0, 4))
        if values.ndim == 1:
            values = values.reshape(1, 4)
        if values.shape[1] != 4 and values.shape[0] == 4:
            values = values.T
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Expected initial states with shape (n, 4).")
        return values

    def plot_linearized_hit_and_run_samples(
        self, samples, projected_mask=None, seeds=None, show=True
    ):
        """Plot sampled initial positions in the obstacle field."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        samples = self._normalize_state_batch(samples)
        fig, ax = plt.subplots(figsize=(9, 7))
        for obstacle in self.obstacles:
            ax.add_patch(Circle(obstacle.center, obstacle.radius, color="0.35", alpha=0.7))
        if samples.size:
            ax.scatter(samples[:, 0], samples[:, 1], c="tab:purple", s=25,
                       label="feasible initial states")
        if seeds is not None:
            seed_values = self._normalize_state_batch(seeds)
            ax.scatter(seed_values[:, 0], seed_values[:, 1], c="black", marker="x", label="seeds")
        ax.scatter(*self.goal, marker="*", s=130, c="gold", edgecolor="black", label="goal")
        ax.set(xlabel="initial x position", ylabel="initial y position")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend()
        if show:
            plt.show()
        return fig, ax

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
        """Plot a two-dimensional projection of point-mass initial states."""
        import matplotlib.pyplot as plt

        values = self._normalize_state_batch(samples if samples is not None else np.empty((0, 4)))
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if feasible_mask is None and values.size:
            feasible_mask = np.ones(values.shape[0], dtype=bool)
        if values.size:
            feasible_mask = np.asarray(feasible_mask, dtype=bool)
            ax.scatter(values[feasible_mask, state_i], values[feasible_mask, state_j],
                       c="tab:green", s=24, label="feasible")
            if np.any(~feasible_mask):
                ax.scatter(values[~feasible_mask, state_i], values[~feasible_mask, state_j],
                           c="tab:red", s=24, label="infeasible")
        ax.set_xlabel(("px", "py", "vx", "vy")[state_i])
        ax.set_ylabel(("px", "py", "vx", "vy")[state_j])
        ax.grid(alpha=0.2)
        if values.size:
            ax.legend()
        if show:
            plt.show()
        return values[feasible_mask] if values.size else values

    def sample_uniform(
        self,
        num_samples: int,
        max_attempts: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Uniformly sample the state box and reject MPC-infeasible draws."""
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")

        feasible_samples: List[np.ndarray] = []
        infeasible_samples: List[np.ndarray] = []
        attempts = 0
        warm_start = self._reference_decision()
        while len(feasible_samples) < num_samples and attempts < max_attempts:
            sample = rng.uniform(self.state_lower, self.state_upper)
            try:
                self._solve_mpc_solution(sample, warm_start)
                feasible_samples.append(sample)
            except RuntimeError:
                infeasible_samples.append(sample)
                # print(f"Attempt {attempts + 1}: Sample {sample} is infeasible.")
            print(f"Feasible samples collected: {len(feasible_samples)} / {num_samples}")
            attempts += 1

        feasible = np.asarray(feasible_samples, dtype=float).reshape(-1, 4)
        infeasible = np.asarray(infeasible_samples, dtype=float).reshape(-1, 4)
        return feasible, infeasible, {
            "requested_samples": int(num_samples),
            "returned_samples": int(feasible.shape[0]),
            "attempts": int(attempts),
            "feasible_candidates": int(feasible.shape[0]),
            "infeasible_candidates": int(infeasible.shape[0]),
        }

    @staticmethod
    def _subtract_interval(
        intervals: List[Tuple[float, float]],
        forbidden: Tuple[float, float],
        tolerance: float = 1e-12,
    ) -> List[Tuple[float, float]]:
        left, right = forbidden
        output: List[Tuple[float, float]] = []
        for low, high in intervals:
            if right <= low + tolerance or left >= high - tolerance:
                output.append((low, high))
                continue
            if left > low + tolerance:
                output.append((low, min(left, high)))
            if right < high - tolerance:
                output.append((max(right, low), high))
        return output

    def _line_intervals(self, z: np.ndarray, direction: np.ndarray) -> List[Tuple[float, float]]:
        linear_interval = self._linear_line_interval(z, direction)
        if linear_interval is None:
            return []
        alpha_low, alpha_high = linear_interval

        intervals = [(alpha_low, alpha_high)]
        positions = self.collision_positions(z)
        position_direction = np.einsum("psj,j->ps", self.collision_position_map, direction)
        for obstacle in self.obstacles:
            delta = positions - np.asarray(obstacle.center)
            radius = obstacle.radius + self.body_radius
            qa = np.einsum("ij,ij->i", position_direction, position_direction)
            qb = 2.0 * np.einsum("ij,ij->i", delta, position_direction)
            qc = np.einsum("ij,ij->i", delta, delta) - radius**2
            for a, b, c in zip(qa, qb, qc):
                discriminant = b * b - 4.0 * a * c
                if a > 1e-15 and discriminant > 0.0:
                    root = np.sqrt(discriminant)
                    intervals = self._subtract_interval(
                        intervals, ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
                    )
                    if not intervals:
                        return []
        return intervals

    def _linear_line_interval(
        self, z: np.ndarray, direction: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        alpha_low, alpha_high = -np.inf, np.inf
        numerator = self.linear_b - self.linear_A @ z
        denominator = self.linear_A @ direction
        for available, slope in zip(numerator, denominator):
            if abs(slope) < 1e-13:
                if available < -1e-8:
                    return None
            elif slope > 0.0:
                alpha_high = min(alpha_high, available / slope)
            else:
                alpha_low = max(alpha_low, available / slope)
        if not np.isfinite(alpha_low) or not np.isfinite(alpha_high) or alpha_low > alpha_high:
            return None
        return alpha_low, alpha_high

    def sample_hit_and_run(
        self,
        num_samples: int,
        max_attempts: int,
        rng: np.random.Generator,
        burn_in: int = 30,
        thinning: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Nonconvex Hit-and-Run using the union of feasible line segments."""
        current = self.feasible_seed()
        accepted: List[np.ndarray] = []
        attempts = 0
        successful_steps = 0
        target_steps = burn_in + thinning * num_samples
        while attempts < max_attempts and successful_steps < target_steps:
            attempts += 1
            direction = rng.normal(size=self.reduced_dimension)
            direction /= np.linalg.norm(direction)
            intervals = self._line_intervals(current, direction)
            lengths = np.asarray([high - low for low, high in intervals])
            valid = np.isfinite(lengths) & (lengths > 1e-11)
            if not np.any(valid):
                continue
            intervals = [interval for interval, keep in zip(intervals, valid) if keep]
            lengths = lengths[valid]
            interval = intervals[rng.choice(len(intervals), p=lengths / lengths.sum())]
            alpha = rng.uniform(interval[0], interval[1])
            candidate = current + alpha * direction
            if not self.is_feasible(candidate, tol=2e-6):
                continue
            current = candidate
            successful_steps += 1
            if successful_steps > burn_in and (successful_steps - burn_in) % thinning == 0:
                accepted.append(current.copy())
        return self._sample_array(accepted), {
            "attempts": attempts,
            "chain_steps": successful_steps,
        }

    def _sample_array(self, samples: List[np.ndarray]) -> np.ndarray:
        if not samples:
            return np.empty((0, self.reduced_dimension))
        return np.asarray(samples, dtype=float).reshape(-1, self.reduced_dimension)


if __name__ == "__main__":
    pm = PointMass()
    
    # Sample random states using the uniform sampler
    # feasible_samples, infeasible_samples, stats = pm.sample_uniform(
    #     num_samples=10,
    #     max_attempts=1000,
    #     rng=np.random.default_rng(42),
    # )
            
    # # Plot the feasible and infeasible samples
    # pm.plot_feasible_set(samples=np.concatenate((feasible_samples, infeasible_samples), axis=0),
    #                      feasible_mask=np.array([True] * len(feasible_samples) + [False] * len(infeasible_samples)),
    #                      show=True)
    
    # Generate feasible samples using the slack-based sampler
    # slack_samples = pm.collect_feasible_samples(
    #     num_samples=10,
    #     max_attempts=1000,
    #     plot_results=False,
    #     slack_penalty="quadratic",
    # )
 
    # Generate feasible samples using the linearized Hit-and-Run sampler
    hit_and_run_samples = pm.collect_linearized_hit_and_run_samples(
        num_samples=10,
        num_feasible_seeds=5,
        max_seed_attempts=1000,
        random_seed=42,
        plot_results=False,
        show=False,
        lp_solver="qpoases",
    )
    
    # Plot the feasible samples generated by the slack-based sampler and the feasible and infeasible samples found from the uniform sampler
    # pm.plot_feasible_set(samples=np.concatenate((feasible_samples, infeasible_samples), axis=0),
    #                      feasible_mask=np.array([True] * len(feasible_samples) + [False] * len(infeasible_samples)),
    #                      show=False)
    # pm.plot_feasible_set(samples=slack_samples,
    #                      feasible_mask=np.array([True] * len(slack_samples)),
    #                      show=False)
    pm.plot_feasible_set(samples=hit_and_run_samples,
                         feasible_mask=np.array([True] * len(hit_and_run_samples)),
                         show=True)
    
    
