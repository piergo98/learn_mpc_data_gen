import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.linalg import solve_discrete_are, solve_continuous_are

# Class implementing a linear system of the form x_{k+1} = A x_k + B u_k and a simple MPC controller
class LinearSystem:
    def __init__(self, A, B, x_bounds, u_bounds, dt=0.1, N=10):
        # Chech the instance of A and B
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        if not isinstance(B, np.ndarray):
            B = np.array(B)
        self.A = A
        self.B = B
        self.dt = dt
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
        self.x_bounds = x_bounds
        self.u_bounds = u_bounds
        
        self.step = self.step_func()
        
        self.good_initial_points = 0
        self.skip = False
        
        self.N = N  # MPC horizon length
        Q = np.eye(self.nx)  # dimension-safe default state cost weight
        # Q = np.diag([30.0, 1.0])  # dimension-safe default state cost weight
        R = np.diag([1.0] * self.nu)  # control cost weight
        self.define_simple_MPC_control(Q, R)
    
    def step_func(self):
        """Compute the next state given current state and control input."""
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)
        x_next = self.A @ x + self.B @ u
        return ca.Function("step", [x, u], [x_next])
    
    def define_simple_MPC_control(self, Q, R):
        ''' A simple MPC controller that tries to stabilize the pole at the upright position. '''
        # Define optimization variables and parameters
        x = ca.SX.sym('x', self.nx)  # state variable
        u = ca.SX.sym('u', self.nu)  # control variable
        x0 = ca.SX.sym('x0', self.nx)  # initial state parameter
        
        X = ca.SX.sym('X', self.nx, self.N+1)  # state trajectory
        U = ca.SX.sym('U', self.nu, self.N)    # control trajectory
        decvar = ca.veccat(X[:,0], ca.vertcat(U, X[:,1:], ))
        # to extract trajectories in nice shape from decvar
        self.extract_traj = ca.Function("extract_traj", [decvar], [X, U])
        self.traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar])
        
        # Define cost function (quadratic cost on state deviation and control effort)
        # Check if Q and R are provided, otherwise use default values
        if Q is None:
            Q = np.diag([1.0] * self.nx)  # state cost weight
        if R is None:
            R = np.diag([1.0] * self.nu)  # control cost weight
            
        E = solve_discrete_are(self.A, self.B, Q, R)
        
        x_next = self.step(x, u)
        l = x.T @ Q @ x + u.T @ R @ u
        
        # Create a CasADi function for the stage cost
        F = ca.Function('F', [x, u], [x_next, l])
        
        # State and control bounds
        self.u_min, self.u_max = -np.array(self.u_bounds), np.array(self.u_bounds)
        self.x_min, self.x_max = -np.array(self.x_bounds), np.array(self.x_bounds)
        
        # Create an NLP to minimize the cost over a horizon
        w = []      # decision variables
        self.lbw = []    # lower bounds on decision variables
        self.ubw = []    # upper bounds on decision variables
        self.w0 = []     # initial guess for decision variables
        J = 0       # objective function
        g = []      # constraints
        self.lbg = []    # lower bounds on constraints
        self.ubg = []    # upper bounds on constraints
        Xk = ca.SX.sym('X0', self.nx)  # initial state variable
        w += [Xk]
        self.lbw += self.x_min.tolist()
        self.ubw += self.x_max.tolist()
        self.w0 += [0.0 for _ in range(self.nx)]
        g += [Xk - x0]  # initial state constraint
        self.lbg += [0.0 for _ in range(self.nx)]
        self.ubg += [0.0 for _ in range(self.nx)]
        for k in range(self.N):
            uk = ca.SX.sym('u_' + str(k), self.nu)
            w += [uk]
            self.lbw += self.u_min.tolist()  # control limits
            self.ubw += self.u_max.tolist()
            self.w0 += [0.0]     # initial guess
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), self.nx)
            w += [Xk]
            self.lbw += self.x_min.tolist()
            self.ubw += self.x_max.tolist()
            self.w0 += [0.0 for _ in range(self.nx)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            self.lbg += [0.0 for _ in range(self.nx)]
            self.ubg += [0.0 for _ in range(self.nx)]
            
            J += l_k  # accumulate cost
            
        # Add terminal constraint
        # g += [Xk]  # terminal state constraint (can be set to a desired target state if needed)
        # self.lbg += [0.0 for _ in range(self.nx)]
        # self.ubg += [0.0 for _ in range(self.nx)]
        J += Xk.T @ ca.DM(E) @ Xk  # terminal cost
        
        # Create an NLP solver instance
        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g), 'p': x0}
        opts = {"ipopt.print_level": 0, "print_time": False, "verbose": False}
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        
    def solve_MPC(self, x0, ret_seq=False, return_traj=False, plot_results=False):
        ''' Solve the MPC problem for a given initial state x0. '''
        sol = self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=x0
        )
        
        x_opt, u_opt = self.extract_traj(sol['x'])
        
        if plot_results:
            x_traj = x_opt.full()
            u_traj = u_opt.full()
            time = np.arange(self.N + 1) * self.dt
            time_u = np.arange(self.N) * self.dt
            
            fig, axes = plt.subplots(self.nx + 1, 1, figsize=(10, 2.5 * (self.nx + 1)))
            for s in range(self.nx):
                ax = axes[s]
                ax.plot(time, x_traj[s, :], linewidth=2)
                ax.set_ylabel(f"x[{s}]", fontsize=10)
                ax.grid(True, alpha=0.3)
                if s == 0:
                    ax.set_title("MPC Trajectory", fontsize=12, fontweight="bold")
            
            # Control subplot (last)
            ax_u = axes[-1]
            if self.nu > 1:
                for j in range(self.nu):
                    ax_u.step(time_u, u_traj[j, :], where='post', linewidth=1.5, label=f"u[{j}]")
                ax_u.legend()
            else:
                ax_u.step(time_u, u_traj[0, :], where='post', linewidth=2, color='k')
            ax_u.set_ylabel('Control', fontsize=10)
            ax_u.set_xlabel('Time (s)', fontsize=10)
            ax_u.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        
        if return_traj:
            return u_opt.full().flatten()[0], x_opt.full(), u_opt.full()
        # Return the optimal control sequence
        if ret_seq:
            return u_opt.full().flatten()  # return the full control sequence as a 1D array
        
        return u_opt.full().flatten()[0]  # return only the first control input
    
    def seek_x0(self, Q, R, plot_results=False):
        # Define optimization variables and parameters
        x = ca.SX.sym('x', self.nx)  # state variable
        u = ca.SX.sym('u', self.nu)  # control variable
        
        # Define cost function (quadratic cost on state deviation and control effort)
        # Check if Q and R are provided, otherwise use default values
        if Q is None:
            Q = np.diag([1.0] * self.nx)  # state cost weight
        if R is None:
            R = np.diag([1.0] * self.nu)  # control cost weight
            
        E = solve_discrete_are(self.A, self.B, Q, R)
        
        x_next = self.step(x, u)
        l = x.T @ Q @ x + u.T @ R @ u
        
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
        
        Xk = ca.SX.sym('X0', self.nx)  # initial state variable
        w += [Xk]
        lbw += [-ca.inf for _ in range(self.nx)]
        ubw += [ca.inf for _ in range(self.nx)]
        w0 += [0.0 for _ in range(self.nx)]
        
        s_x0 = ca.SX.sym('slack_x0', self.nx, 2)  # slack variables for initial state bounds
        s_x0_var = ca.vec(s_x0)  # flatten the slack variables for easier handling in the cost
        w += [s_x0_var]
        lbw += [0.0 for _ in range(self.nx * 2)]
        ubw += [ca.inf for _ in range(self.nx * 2)]
        w0 += [0.0 for _ in range(self.nx * 2)]
        
        # Initial state bounds with slack
        g += [Xk - self.x_max + s_x0[:, 0], -Xk + self.x_min + s_x0[:, 1]]  
        lbg += [0.0 for _ in range(self.nx * 2)]
        ubg += [0.0 for _ in range(self.nx * 2)]
        
        X_syms = [Xk]
        U_syms = []
        slacks = [s_x0_var]
        
        for k in range(self.N):
            uk = ca.SX.sym('u_' + str(k), self.nu)
            w += [uk]
            lbw += [-ca.inf for _ in range(self.nu)] # no control limits
            ubw += [ca.inf for _ in range(self.nu)]  # no control limits
            w0 += [0.0 for _ in range(self.nu)]     # initial guess
            
            s_uk = ca.SX.sym('slack_u_' + str(k), self.nu, 2)  # slack variables for control bounds
            s_uk_var = ca.vec(s_uk)  # flatten the slack variables for easier handling in the cost
            w += [s_uk_var]
            lbw += [0.0 for _ in range(self.nu * 2)]
            ubw += [ca.inf for _ in range(self.nu * 2)]
            w0 += [0.0 for _ in range(self.nu * 2)]
            
            g += [uk - self.u_max + s_uk[:, 0], -uk + self.u_min + s_uk[:, 1]]
            lbg += [0.0 for _ in range(self.nu * 2)]
            ubg += [0.0 for _ in range(self.nu * 2)]
            
            Xk_next, l_k = F(Xk, uk)
            
            Xk = ca.SX.sym('X_' + str(k+1), self.nx)
            w += [Xk]
            lbw += [-ca.inf for _ in range(self.nx)]
            ubw += [ca.inf for _ in range(self.nx)]
            w0 += [0.0 for _ in range(self.nx)]
            
            s_xk = ca.SX.sym('slack_x_' + str(k), self.nx, 2)  # slack variables for state bounds
            s_xk_var = ca.vec(s_xk)  # flatten the slack variables for easier handling in the cost
            w += [s_xk_var]
            lbw += [0.0 for _ in range(self.nx * 2)]
            ubw += [ca.inf for _ in range(self.nx * 2)]
            w0 += [0.0 for _ in range(self.nx * 2)]
            
            g += [Xk - self.x_max + s_xk[:, 0], -Xk + self.x_min + s_xk[:, 1]]  # state bounds with slack
            lbg += [0.0 for _ in range(self.nx * 2)]
            ubg += [0.0 for _ in range(self.nx * 2)]
            
            g += [Xk_next - Xk]  # dynamics constraint
            lbg += [0.0 for _ in range(self.nx)]
            ubg += [0.0 for _ in range(self.nx)]
                
            J += l_k
            X_syms.append(Xk)
            U_syms.append(uk)
            slacks.append(s_uk_var)
            slacks.append(s_xk_var)
            
        terminal_value = Xk.T @ ca.DM(E) @ Xk
        alpha = 1e0
        J += terminal_value
        g += [terminal_value]
        lbg += [-ca.inf]
        ubg += [alpha]
        
        # Convert the list of slacks into a single vector for easier handling in the cost function
        slacks_vec = ca.vertcat(*slacks)
        
        # Add a quadratic penalty on the slack variables to encourage them to be small and a small regularization term
        # max_spikes = max(1, int(0.15 * slacks_vec.shape[0]))
        # print(max_spikes)
        # raise NotImplementedError("The spiky array generation is currently disabled. Please implement it if you want to use this feature.")
        rho, _ = self.spiky_array(
            slacks_vec.shape[0], 
            min_spikes=1, 
            max_spikes=12, 
            base_scale=1.0, 
            spike_factor_range=(100, 1000), 
            positive=True
        )
        
        Q_s = np.diag(rho)
        J += ca.vec(slacks_vec).T @ Q_s @ ca.vec(slacks_vec) + 1e-6 * ca.dot(slacks_vec, slacks_vec)
        
        X_stack = ca.hcat(X_syms)
        U_stack = ca.hcat(U_syms)
        extract = ca.Function("extract_extreme_traj", [ca.vertcat(*w)], [X_stack, U_stack])
        extract_slacks = ca.Function("extract_slacks", [ca.vertcat(*w)], slacks)

        nlp_prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
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
        S = extract_slacks(sol['x'])
        
        x0_opt = X_opt[:, 0].full().flatten()
        u0_opt = float(U_opt[:, 0])
        return x0_opt, u0_opt, return_status
    
    def spiky_array(self, n, min_spikes=1, max_spikes=3, base_scale=1.0, spike_factor_range=(50, 1000), positive=True):
        """Generate an array of length n with a few large 'spikes' and the rest small values.
        
        Parameters
        ----------
        n : int
            Length of the array.
        min_spikes : int, optional
            Minimum number of large components (default: 1).
        max_spikes : int, optional
            Maximum number of large components (default: 3).
        base_scale : float, optional
            Scale parameter for the background components (default: 1.0).
        spike_factor_range : tuple, optional
            Range of factors for the spike components (default: (50, 1000)).
        positive : bool, optional
            Whether to generate only positive values (default: True).

        Returns
        -------
        x : np.ndarray
            Generated array.
        spike_idx : np.ndarray
            Indices of the spike components.
        """
        # small/background components
        x = np.random.exponential(base_scale, size=n) if positive else np.random.normal(0, base_scale, size=n)

        # choose how many big components
        k = np.random.randint(min_spikes, min(max_spikes, n) + 1)

        # choose spike locations
        spike_idx = np.random.choice(n, size=k, replace=False)

        # make them much bigger
        factors = np.random.uniform(*spike_factor_range, size=k)
        x[spike_idx] *= factors

        return x, spike_idx

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
            Number of simulation steps (default: 60).
        plot_results : bool, optional
            Whether to plot the results (default: True).
        
        Returns
        -------
        x_traj : np.ndarray
            State trajectory of shape (N+1, 4).
        u_traj : np.ndarray
            Control trajectory of shape (N,).
        """
        # Initialize trajectories
        x_traj = np.zeros((Nsim + 1, self.nx))
        u_traj = np.zeros((Nsim, self.nu))
        
        x_traj[0] = np.array(x0).flatten()
        
        # Simulate forward in time
        for k in range(Nsim):
            # Get control input
            if control_policy is None:
                u_k = self.solve_MPC(x_traj[k])  # Solve MPC to get control input for current state
            else:
                u_k = control_policy(x_traj[k])  # Get control input from the provided policy function
            
            # Ensure u_k is scalar
            if hasattr(u_k, '__len__'):
                u_k = float(u_k[0])
            else:
                u_k = float(u_k)
            
            u_traj[k, :] = u_k
            
            # Step forward
            x_next = self.step(x_traj[k], [u_k])
            x_traj[k + 1] = np.array(x_next.full()).flatten()
        
        if control_policy is None:
            if self.solver.stats()['success']:
                self.good_initial_points += 1
            else:
                # print(self.solver.stats()['return_status'])
                self.skip = True
        
        if not plot_results:
            return x_traj, u_traj
        
        # Plot results (dynamic number of state plots based on self.nx)
        time = np.arange(Nsim + 1) * self.dt
        time_u = np.arange(Nsim) * self.dt

        n_plots = max(1, self.nx) + 1  # one subplot per state, plus one for control
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        title = "Linear System Simulation"

        # Plot each state in its own subplot
        for s in range(self.nx):
            ax = axes[s]
            ax.plot(time, x_traj[:, s], linewidth=2)
            ax.set_ylabel(f"x[{s}]", fontsize=10)
            ax.grid(True, alpha=0.3)
            if s == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")

        # Control subplot (last)
        ax_u = axes[-1]
        # If control is multi-dimensional, plot the first channel
        if self.nu > 1:
            for j in range(self.nu):
                ax_u.step(time_u, u_traj[:, j], where='post', linewidth=1.5, label=f"u[{j}]")
            ax_u.legend()
        else:
            ax_u.step(time_u, u_traj[:, 0], where='post', linewidth=2, color='k')
        ax_u.set_ylabel('Control', fontsize=10)
        ax_u.set_xlabel('Time (s)', fontsize=10)
        ax_u.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        return x_traj, u_traj

    def plot_feasible_set(
        self,
        state_i=0,
        state_j=1,
        samples=None,
        feasible_mask=None,
        check_mpc=True,
        uniform_samples=None,
        uniform_seed=42,
        show=True,
        ax=None,
    ):
        """Visualize the state bounds as a feasible region in 2D."""
        bounds = np.array(self.x_bounds, dtype=float).reshape(-1)
        if bounds.size < max(state_i, state_j) + 1:
            raise ValueError("state_i/state_j out of range for x_bounds.")
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

        if uniform_samples is not None:
            rng = np.random.default_rng(uniform_seed)
            num_uniform = int(uniform_samples)
            if num_uniform < 1:
                raise ValueError("uniform_samples must be a positive integer.")
            uniform_points = rng.uniform(
                low=-bounds,
                high=bounds,
                size=(num_uniform, self.nx),
            )
            if check_mpc:
                uniform_mask = np.array(
                    [self.is_feasible(x0, Nsim=30) for x0 in uniform_points],
                    dtype=bool,
                )
            else:
                uniform_mask = np.all(np.abs(uniform_points) <= bounds, axis=1)

            ax.scatter(
                uniform_points[uniform_mask, state_i],
                uniform_points[uniform_mask, state_j],
                s=20,
                color="g",
                alpha=1.0,
                label="Uniform feasible",
            )
            if np.any(~uniform_mask):
                ax.scatter(
                    uniform_points[~uniform_mask, state_i],
                    uniform_points[~uniform_mask, state_j],
                    s=20,
                    color="r",
                    alpha=0.6,
                    label="Uniform infeasible",
                )

        if samples is not None:
            samples_arr = np.asarray(samples, dtype=float)
            if samples_arr.ndim == 3:
                samples_arr = samples_arr.reshape(-1, samples_arr.shape[-1])
            if samples_arr.ndim == 2 and samples_arr.shape[0] == self.nx:
                samples_arr = samples_arr.T
            if samples_arr.ndim != 2 or samples_arr.shape[1] != self.nx:
                raise ValueError("samples must have shape (N, nx) or (nx, N).")

            if feasible_mask is None and check_mpc:
                feasible_mask = np.array([self.is_feasible(x0, Nsim=30) for x0 in samples_arr], dtype=bool)
            elif feasible_mask is None:
                feasible_mask = np.all(np.abs(samples_arr) <= bounds, axis=1)
            else:
                feasible_mask = np.asarray(feasible_mask, dtype=bool)
                if feasible_mask.shape[0] != samples_arr.shape[0]:
                    raise ValueError("feasible_mask length must match number of samples.")

            print(f"Feasible samples: {np.sum(feasible_mask)} / {len(samples_arr)}")

            ax.scatter(
                samples_arr[feasible_mask, state_i],
                samples_arr[feasible_mask, state_j],
                s=20,
                color="tab:cyan",
                alpha=0.4,
                label="Our strategy feasible",
            )
            if np.any(~feasible_mask):
                ax.scatter(
                    samples_arr[~feasible_mask, state_i],
                    samples_arr[~feasible_mask, state_j],
                    s=30,
                    color="tab:black",
                    alpha=0.4,
                    label="Our strategy infeasible",
                )

        ax.set_xlabel(f"x[{state_i}]")
        ax.set_ylabel(f"x[{state_j}]")
        # ax.set_xlabel(r"$\theta$ (rad)", fontsize=12)
        # ax.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=12)
        # ax.set_title("Feasible Set (State Bounds)")
        ax.set_xlim(x_min * 1.05, x_max * 1.05)
        ax.set_ylim(y_min * 1.05, y_max * 1.05)
        ax.grid(True, alpha=0.3)
        if samples is not None or uniform_samples is not None:
            ax.legend(loc="best", fontsize=15)

        if show:
            plt.tight_layout()
            plt.show()

        return samples_arr[feasible_mask, :]

    def collect_feasible_samples(self, num_samples=100, max_attempts=1000, plot_results=True, uniform_samples=None):
        """Collect feasible initial-state samples for the MPC problem."""
        if num_samples < 1:
            x0_samples = None
        else:
            samples = []
            for attempt in range(max_attempts):
                if len(samples) >= num_samples:
                    break
                x0_opt, u0_opt, return_status = self.seek_x0(Q=None, R=None, plot_results=False)
                if return_status != "Solve_Succeeded":
                    print(
                        f"Attempt {attempt + 1}: solver failed (status: {return_status}), skipping..."
                    )
                    continue

                samples.append((x0_opt, u0_opt))
                print(f"Collected {len(samples)}/{num_samples}")

            if len(samples) < num_samples:
                print(f"Only collected {len(samples)} samples after {max_attempts} attempts")
            else:
                print(f"Collected {num_samples} samples")

            print("Sample samples (x0, u0):")
            for i, (x0_sol, u0_sol) in enumerate(samples[:10]):
                print(f"{i + 1}: x0 = {x0_sol}, u0 = {u0_sol}")

            x0_samples = np.array([x0 for x0, _u0 in samples], dtype=float)
        if plot_results:
            feas_samples = self.plot_feasible_set(
                samples=x0_samples,
                check_mpc=True,
                uniform_samples=uniform_samples,
                show=True,
            )
        else:
            feasible_mask = np.array([self.is_feasible(x0) for x0 in x0_samples], dtype=bool)
            feas_samples = x0_samples[feasible_mask]
            
        return feas_samples

    def generate_training_data(self):
        """Generate training data by simulating the system from random initial states."""
        Ncsim = 100
        N = 10
        alpha = 0.3
        p_bound = 1.5 * alpha
        v_bound = 1.0 * alpha 
        theta_bound = 1.0 * alpha
        omega_bound = 0.35 * alpha
        np.random.seed(36)  # for reproducibility
        
        # For each simulation, store 
        training_initial_states = []
        training_first_control = []
        training_full_controls = []
        
        for _ in range(Ncsim):
            p0 = np.random.uniform(-p_bound, p_bound)
            v0 = np.random.uniform(-v_bound, v_bound)
            theta0 = np.random.uniform(-theta_bound, theta_bound)
            omega0 = np.random.uniform(-omega_bound, omega_bound)
            x0 = [theta0, p0, omega0, v0]
            
            # Simulate forward in time
            u_traj = np.zeros((N, self.nu))
            for k in range(N):
                training_initial_states.append(x0)
                # Get control input
                u_k = self.solve_MPC(x0)  # Solve MPC to get control input for current state
                training_first_control.append(u_k)
                
                # Ensure u_k is scalar
                if hasattr(u_k, '__len__'):
                    u_k = float(u_k[0])
                else:
                    u_k = float(u_k)
                
                u_traj[k, :] = u_k
                
                # Step forward
                x_next = self.step(x0, [u_k])
                x0 = np.array(x_next.full()).flatten()
            
                if self.solver.stats()['success']:
                    self.good_initial_points += 1
                else:
                    # print(self.solver.stats()['return_status'])
                    self.skip = True
            training_full_controls.append(u_traj.flatten())
            
        # Save training data in two csv files so that we can use it later for supervised learning
        # from pathlib import Path
        # import pandas as pd
        # path = Path(__file__).parent / 'data'
        # print(path)
        
        # df_1d = pd.DataFrame({
        #     'initial_state': training_initial_states,
        #     'first_control': training_first_control,
        # })
        # df_1d.to_csv(path / 'linear_system_training_data_1d.csv', index=False)
        
        # df_Nd = pd.DataFrame({
        #     # only keep the initial state for each simulation
        #     'initial_state': training_initial_states[::N],  
        #     'full_control_sequence': training_full_controls,
        # })
        # df_Nd.to_csv(path / 'linear_system_training_data_Nd.csv', index=False)
        
        # print(f"Generated training data for {Ncsim} simulations")
    
if __name__ == "__main__":
    # Define A and B matrices
    A = [[1.0, 0.1],
        [0.981, 1.0]]
    B = [[0.0], [0.1]]
    # A = [[1.0, 0.1, 0, 0],
    #      [0, 0.9818, 0.2673, 0],
    #      [0, 0, 1.0, 0.1],
    #      [0, -0.0455, 3.1182, 1.0]]
    # B = [[0.0], [0.1818], [0.0], [0.4546]]
    
    # Create linear system instance
    x_bounds= np.array([2.5, 3.5], dtype=float)
    u_bounds= np.array([2.0], dtype=float)
    # x_bounds= np.array([1.0, 1.5, 0.35, 1.0], dtype=float)
    # u_bounds= np.array([1.0], dtype=float)
    lin_sys = LinearSystem(A, B, x_bounds, u_bounds, dt=0.1, N=10)
    # Ncsim = 200
    
    # Generate training data for supervised learning
    # lin_sys.generate_training_data()
     
    # Close-loop simulation with MPC control
    # lin_sys.close_loop_simulation(x0, plot_results=True)
    # training_initial_states = np.random.uniform(-x_lim, x_lim, size=(1000, lin_sys.nx))
    # lin_sys.plot_feasible_set(state_i=0, state_j=1, samples=training_initial_states, check_mpc=True)
    feasible_samples = lin_sys.collect_feasible_samples(num_samples=100, max_attempts=500, plot_results=True, uniform_samples=None)
    # lin_sys.solve_MPC(feasible_samples[0], plot_results=True)   
    # lin_sys.close_loop_simulation(feasible_samples[0], plot_results=True)
    # print(f"Number of good initial points: {lin_sys.good_initial_points} out of {Ncsim}")
