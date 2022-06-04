import numpy as np

class Thruster:
    def __init__(self, position, orientation, thrust):
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)
        self.thrust = thrust

    def control_to_torque(self, u):
        return np.cross(self.position, u*self.thrust*self.orientation)

    def __repr__(self):
        return f'Thruster:\n position: {self.position}\n orientation: {self.orientation}\n thrust: {self.thrust}\n'