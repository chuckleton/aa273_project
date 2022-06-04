import numpy as np
import transformations
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import sqrtm


def x_tilde_dot(t, x_tilde, J, u):
    u_tilde = transformations.forward_u_transformation(x_tilde, u, J)
    return np.concatenate((x_tilde[3:], u_tilde))

class Body:
    def __init__(self, q0, omega0, J):
        self.q0 = q0
        self.omega0 = omega0
        self.J = J
        self.q_base = Rot.from_quat(q0)
        self.x0 = np.concatenate((q0[:3], omega0))
        self.x_tilde0 = transformations.forward_x_transformation(self.x0, self.J)

        self.x_tilde = self.x_tilde0

        self.true_state = (self.q_base.as_quat(), self.omega0)

    def step(self, u, Qs, C, R, delta_t = 0.1):
        ts = np.linspace(0, delta_t, 2)
        res = solve_ivp(x_tilde_dot, [0, delta_t], y0=self.x_tilde, t_eval=ts, args=(
            self.J, u), method='DOP853', rtol=1e-5, atol=1e-8)
        self.x_tilde = res.y[:, -1]
        x = transformations.inverse_x_transformation(
            self.x_tilde) + delta_t*sqrtm(Qs)@np.random.randn(6)
        self.x_tilde = transformations.forward_x_transformation(x, self.J)
        q = np.append(self.x_tilde[:3], np.sqrt(
            1 - np.sum(self.x_tilde[:3]**2)))
        self.q_base = Rot.from_quat(q)*self.q_base
        self.true_state = self.get_true_state()
        # meas = C@self.x_tilde
        # meas[:3] = self.true_state[0][:3]
        meas = C@self.x_tilde + sqrtm(R)@np.random.randn(C.shape[0])
        meas = meas + sqrtm(R)@np.random.randn(C.shape[0])
        self.x_tilde[:3] = np.zeros(3)
        self.x_tilde[3:] = 0.5*self.J@x[3:]
        return meas

    def get_true_state(self):
        q = self.q_base.as_quat()
        omega = transformations.inverse_x_transformation(self.x_tilde)[3:]
        return q, omega
