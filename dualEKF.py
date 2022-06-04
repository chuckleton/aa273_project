import numpy as np
from scipy.spatial.transform import Rotation as Rot

import transformations

class EKFState:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

class DualEKFState:
    def __init__(self, state_s, state_p):
        self.s = state_s
        self.p = state_p

    @property
    def mu(self):
        return self.s.mu

    @mu.setter
    def mu(self, value):
        self.s.mu = value

    @property
    def sigma(self):
        return self.s.sigma

    @sigma.setter
    def sigma(self, value):
        self.s.sigma = value

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

class DualEKF:
    def __init__(self, initial_state, J, R, Qs, Qp, T):
        self.state = initial_state
        self.J = J
        self.R = R
        self.Qs = Qs
        self.Qp = Qp
        self.T = T
        self.M = T.shape[1]

        q0 = np.sqrt(1-np.dot(initial_state.s.mu[:3], initial_state.s.mu[:3]))
        self.q_base = Rot.from_quat(np.append(initial_state.s.mu[:3], q0))

        self.A = np.zeros((6,6))
        self.A[:3,3:] = np.eye(3)
        self.A[3:,3:] = np.eye(3)
        self.B = np.zeros((6,3))
        self.B[3:] = np.eye(3)
        self.C = np.eye(6)

        self.partial_x_prev = np.zeros((6, self.M))
        self.partial_g_prev = np.zeros((6, self.M))
        self.K_prev = np.zeros((6, 6))
        self.R_prev = np.zeros((6, 6))

    def controls_to_torque(self, us):
        torque = self.T@(us*self.state.p.mu)
        return torque

    def predict_parameters(self, u, delta_t):
        Q = self.Qp*delta_t**2
        self.state.p.sigma += Q

    def predict_state(self, u, delta_t):
        A = self.A.copy()
        A[:3] *= delta_t
        B = self.B*delta_t
        Q = self.Qs*delta_t**2
        u_tilde = transformations.forward_u_transformation(
            self.state.s.mu,
            self.controls_to_torque(u),
            self.J)
        self.state.s.mu = A@self.state.s.mu + B@u_tilde
        self.state.s.sigma = A@self.state.s.sigma@A.T + Q

    def partial_u_tilde_partial_theta(self, x, u):
        q = x[:3]
        q0 = np.sqrt(1-np.dot(q,q))
        partial_omega_dot = np.linalg.inv(self.J)@self.T@np.diag(u)
        partial_u_tilde = 0.5*(q0*np.eye(3) + skew(q))@partial_omega_dot
        return partial_u_tilde

    def G_tot(self, u, delta_t):
        partial_u_tilde = self.partial_u_tilde_partial_theta(self.state.s.mu, u)
        partial_x_prev = self.R_prev@(self.partial_x_prev - self.K_prev@self.partial_g_prev)
        partial_x = delta_t*(self.B@partial_u_tilde +
                             self.A@partial_x_prev)
        partial_g = self.C@partial_x
        G_tot = partial_g
        self.partial_x_prev = partial_x
        self.partial_g_prev = partial_g
        return G_tot

    def update_parameters(self, y, u, delta_t):
        R = self.R
        P = self.state.p.sigma
        G_tot = self.G_tot(u, delta_t)
        Kp = P@G_tot.T@np.linalg.inv(G_tot@P@G_tot.T + R)
        self.state.p.mu += Kp@(y - self.C@self.state.s.mu)
        I = np.eye(self.M)
        self.state.p.sigma = (I - Kp@G_tot)@P@(I - Kp@G_tot).T + Kp@R@Kp.T

    def update_state(self, y, u, delta_t):
        P = self.state.s.sigma
        G = self.C
        R = self.R

        nu = y - self.C@self.state.s.mu
        S = G@P@G.T + R

        Ks = P@G.T@np.linalg.inv(S)
        self.K_prev = Ks
        self.state.s.mu += Ks@nu
        I = np.eye(6)
        self.state.s.sigma = (I - Ks@G)@P@(I - Ks@G).T + Ks@R@Ks.T
        return nu, S

    def predict(self, u, delta_t):
        self.predict_parameters(u, delta_t)
        self.predict_state(u, delta_t)

    def update(self, y, u, delta_t):
        self.update_parameters(y, u, delta_t)
        return self.update_state(y, u, delta_t)

    def step(self, y, u, delta_t = 0.1):
        self.predict(u, delta_t)
        self.update(y, u, delta_t)
        q0 = np.sqrt(1-np.dot(self.state.s.mu[:3], self.state.s.mu[:3]))
        q_new = np.append(self.state.s.mu[:3], q0)
        rotation = Rot.from_quat(q_new)
        self.q_base = rotation*self.q_base
        R_prev = rotation.inv().as_matrix()
        self.R_prev[:3,:3] = R_prev
        self.R_prev[3:,3:] = R_prev
        x = transformations.inverse_x_transformation(self.state.s.mu)
        self.state.s.mu[:3] = np.zeros(3)
        self.state.s.mu[3:] = 0.5*self.J@x[3:]

    def get_true_state(self):
        q = self.q_base.as_quat()
        omega = transformations.inverse_x_transformation(self.state.s.mu)[3:]
        return q, omega