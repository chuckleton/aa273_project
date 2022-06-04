import numpy as np
from scipy.spatial.transform import Rotation as Rot

import transformations


class EKFState:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class EKF:
    def __init__(self, initial_state, J, R, Qs, T, theta):
        self.state = initial_state
        self.J = J
        self.R = R
        self.Qs = Qs
        self.T = T
        self.M = T.shape[1]
        self.theta = theta

        q0 = np.sqrt(1-np.dot(initial_state.mu[:3], initial_state.mu[:3]))
        self.q_base = Rot.from_quat(np.append(initial_state.mu[:3], q0))

        self.A = np.zeros((6, 6))
        self.A[:3, 3:] = np.eye(3)
        self.A[3:, 3:] = np.eye(3)
        self.B = np.zeros((6, 3))
        self.B[3:] = np.eye(3)
        self.C = np.eye(6)

    def controls_to_torque(self, us):
        torque = self.T@(us*self.theta)
        return torque

    def predict_state(self, u, delta_t):
        A = self.A
        A[:3] *= delta_t
        B = self.B*delta_t
        Q = self.Qs*delta_t**2
        u_tilde = transformations.forward_u_transformation(
            self.state.mu, self.controls_to_torque(u), self.J)
        self.state.mu = A@self.state.mu + B@u_tilde
        self.state.sigma = A@self.state.sigma@A.T + Q

    def update_state(self, y, u, delta_t):
        P = self.state.sigma
        G = self.C
        R = self.R

        nu = y - self.C@self.state.mu
        S = G@P@G.T + R

        Ks = P@G.T@np.linalg.inv(S)
        self.state.mu += Ks@nu
        I = np.eye(6)
        self.state.sigma = (I - Ks@G)@P
        return nu, S

    def predict(self, u, delta_t):
        self.predict_state(u, delta_t)

    def update(self, y, u, delta_t):
        return self.update_state(y, u, delta_t)

    def step(self, y, u, delta_t=0.1):
        self.predict(u, delta_t)
        self.update(y, u, delta_t)
        q0 = np.sqrt(1-np.dot(self.state.s.mu[:3], self.state.s.mu[:3]))
        q_new = np.append(self.state.s.mu[:3], q0)
        rotation = Rot.from_quat(q_new)
        self.q_base = rotation*self.q_base
        R_prev = rotation.inv().as_matrix()
        self.R_prev[:3, :3] = R_prev
        self.R_prev[3:, 3:] = R_prev
        x = transformations.inverse_x_transformation(self.state.s.mu)
        self.state.s.mu[:3] = np.zeros(3)
        self.state.s.mu[3:] = 0.5*self.J@x[3:]

    def get_true_state(self):
        q = self.q_base.as_quat()
        omega = transformations.inverse_x_transformation(self.state.s.mu)[3:]
        return q, omega
