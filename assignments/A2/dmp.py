from this import d
import numpy as np
import scipy.interpolate
import pickle

class DMP(object):
    """
	Dynamic Movement Primitives wlearned by Locally Weighted Regression (LWR).

    Implementation of P. Pastor, H. Hoffmann, T. Asfour and S. Schaal, "Learning and generalization of
    motor skills by learning from demonstration," 2009 IEEE International Conference on Robotics and
    Automation, 2009, pp. 763-768, doi: 10.1109/ROBOT.2009.5152385.
	"""

    def __init__(self, nbasis=30, K_vec=10*np.ones((6,)), weights=None):
        self.nbasis = nbasis  # Basis function number
        self.K_vec = K_vec
        
        self.K = np.diag(self.K_vec)  # Spring constant
        self.D = np.diag(2 * np.sqrt(self.K_vec))  # Damping constant, critically damped

        # used to determine the cutoff for s
        self.convergence_rate = 0.01
        self.alpha = -np.log(self.convergence_rate)

        # Creating basis functions and psi_matrix
        # Centers logarithmically distributed between 0.001 and 1
        self.basis_centers = np.logspace(-3, 0, num=self.nbasis)
        self.basis_variances = self.nbasis / (self.basis_centers ** 2)

        self.weights = weights

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(
                nbasis=self.nbasis,
                K_vec=self.K_vec,
                weights=self.weights,
            ), f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        dmp = cls(
            nbasis=data["nbasis"],
            K_vec=data["K_vec"],
            weights=data["weights"]
        )
        return dmp

    def learn(self, X, T):
        """
        Learn the weights of the DMP using Locally Weighted Regression.

        X: demonstrated trajectories. Has shape [number of demos, number of timesteps,  dofs].
        T: corresponding timings. Has shape [number of demos, number of timesteps].
            It is assumed that trajectories start at t=0
        """
        # 
        num_demos = X.shape[0]
        num_timesteps = X.shape[1]

        # Initial position : [num_demos, num_timesteps, num_dofs]
        x0 = np.tile(X[:, 0, :][:, None, :], (1, num_timesteps, 1))
        # Goal position : [num_demos, num_timesteps, num_dofs]
        g = np.tile(X[:, -1, :][:, None, :], (1, num_timesteps, 1))
         # Duration of the demonstrations
        tau = T[:, -1] 

        tau = tau[:, None, None]
        # TODO: Compute s(t) for each step in the demonstrations
        # tau = np.reshape(tau, (tau.shape[0], 1))
        s = np.exp((-self.alpha / tau) * T)


        # TODO: Compute x_dot and x_ddot using numerical differentiation (np.graident)
        x_dot = np.gradient(X, axis=1)
        x_ddot = np.gradient(x_dot, axis=1)

        # TODO: Temporal Scaling by tau.
        # v_dot = (self.K_vec * (g-X) - d*x_dot + (g-x0)*fs) / tau
        v_dot = tau * x_ddot
        # TODO: Compute f_target(s) based on Equation 8.
        # print(f"Shape of k, g, X, D, tau: {(self.K_vec.shape, g.shape, X.shape, self.D.shape, tau.shape)}")
        # f_s_target = (-self.K_vec* (g-X) + self.D*x_dot + tau*v_dot) / g - x0
        # f_s_target = []
        # for i in range(X.shape[1]):
        #     x = np.tile(X[:, i, :][:, None, :], (1, num_timesteps, 1))
        #     x_dot_i = np.tile(x_dot[:, i, :][:, None, :], (1, num_timesteps, 1))
        #     f_s_target.append()

        f_s_target = (tau * v_dot + np.diagonal(self.D) * x_dot) / self.K_vec - (g-X) + (g-x0)*s

        print(f"Shape of s, var, centers: {(s.shape, self.basis_variances.shape, self.basis_centers.shape)}")

        # TODO: Compute psi(s). Hint: shape should be [num_demos, num_timesteps, nbasis]
        psi = np.exp(-self.basis_variances[:, None, None] * (s - self.basis_centers[:, None, None])**2)

        # TODO: Solve a least squares problem for the weights.
        # Hint: minimize f_target(s) - f_w(s) wrt to w
        # Hint: you can use np.linalg.lstsq
        if not self.weights:
            self.weights = np.ones(num_demos)
        f_ws = np.sum(self.weights*psi*s) / np.sum(psi)

        self.weights, _ = np.linalg.lstsq(f_ws, f_s_target)


    def execute(self, t, dt, tau, x0, g, x_t, xdot_t):
        """
        Query the DMP at time t, with current position x_t, and velocity xdot_t.
        The parameter tau controls temporal scaling, x0 sets the initial position
        and g sets the goal for the trajectory.

        Returns the next position x_{t + dt} and velocity x_{t + dt}
        """
        if self.weights is None:
            raise ValueError("Cannot execute DMP before parameters are set by DMP.learn()")

        # Calculate s(t) by integrating 
        s = np.exp(((-self.alpha / tau) * t))

        # TODO: Compute f(s). See equation 3.
        psi = np.exp(-self.basis_variances * (s - self.basis_centers)**2)
        f_s = np.sum(self.weights*psi*s) / np.sum(psi)

        # Temporal Scaling
        v_t = tau * xdot_t

        # TODO: Calculate acceleration. Equation 6
        v_dot_t = (self.K_vec*(g-x_t) - dt*v_t - self.K_vec*(g-x0)*s + self.K_vec*f_s) / tau

        # TODO: Calculate next position and velocity
        xdot_tp1 = v_t + v_dot_t*dt
        x_tp1 = x_t + v_t*dt

        return x_tp1, xdot_tp1

    def rollout(self, dt, tau, x0, g):
        time = 0
        x = x0
        x_dot = np.zeros_like(x0)
        X = [x0]
      
        while time <= tau:
            x, x_dot = self.execute(t=time, dt=dt, tau=tau, x0=x0, g=g, x_t=x, xdot_t=x_dot)
            time += dt
            X.append(x)

        return np.stack(X)


    def _interpolate(self, trajectories, initial_dt):
        """
        Combine the given variable length trajectories into a fixed length array
        by interpolating shorter arrays to the maximum given sequence length.

        trajectories: A list of N arrays of shape (T_i, num_dofs) where T_i is the number
            of time steps in trajectory i
        initial_dt: A scalar corresponding to the duration of each time step.

        Returns: A numpy array of shape (N, max_i T_i, num_dofs)
        """
        length = max(len(traj) for traj in trajectories)
        dofs = trajectories[0].shape[1]

        X = np.zeros((len(trajectories), length, dofs))
        T = np.zeros((len(trajectories), length))
        
        for ti, traj in enumerate(trajectories):
            t = np.arange(len(traj)) * initial_dt
            t_new = np.linspace(0, t.max(), length)
            T[ti, :] = t_new
            for deg in range(dofs):
                path_gen = scipy.interpolate.interp1d(t, traj[:,deg])
                X[ti, :, deg] = path_gen(t_new)
        return X, T
