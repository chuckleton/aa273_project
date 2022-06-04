import numpy as np

from body import Body

class Spacecraft:
    def __init__(self, q0, omega0, J, thrusters, C):
        self.q0 = q0
        self.omega0 = omega0
        self.body = Body(q0, omega0, J)
        self.J = J
        self.C = C
        self.thrusters = thrusters
        self.num_thrusters = len(thrusters)
        self.generate_torque_matrix()

    def generate_torque_matrix(self):
        self.T = np.empty((3, self.num_thrusters))
        u = np.ones(3)
        for i, thruster in enumerate(self.thrusters):
            self.T[:,i] = thruster.control_to_torque(u)

    def controls_to_torque(self, us, thetas):
        torque = self.T@(us*thetas)
        return torque

    def step(self, us, thetas, Qs, R, delta_t=0.1):
        u = self.controls_to_torque(us, thetas)
        meas = self.body.step(u, Qs, self.C, R, delta_t)
        return meas

    def step_direct(self, us, delta_t=0.1):
        Qs = np.zeros((6, 6))
        Qs[3:,3:] = np.eye(3)*0.0
        R = np.eye(6)*0.000
        meas = self.body.step(us, Qs, self.C, R, delta_t)
        return meas

    def get_true_state(self):
        return self.body.get_true_state()

    def __repr__(self):
        state = self.get_true_state()
        return f'Spacecraft:\n state: (q: {state[0]}, omega: {state[1]})\n Number of thrusters: {self.num_thrusters}\n, J: {self.J}\n'
