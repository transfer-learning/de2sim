import numpy as np


class State:
    def __init__(self, pose=np.asmatrix([0.0, 0.0, 0.0]).T, twist=np.asmatrix([0.0, 0.0]).T):
        self.pose = pose
        self.twist = twist


class DE2Config:
    def __init__(self, axle_length: float = 0.15, drag_coeff: float = 1.0):
        self.axle_length = axle_length
        self.drag_coeff = drag_coeff

    def __str__(self):
        return f"axle_length: {self.axle_length}\ndrag_coeff: {self.drag_coeff}"

    def __repr__(self):
        return f"axle_length: {self.axle_length}\ndrag_coeff: {self.drag_coeff}"


class DE2Bot:
    def __init__(self, state: State = State(), config: DE2Config = DE2Config()):
        self.state = state
        self.config = config

        self.G = np.asmatrix([
            [1.0, 1.0],
            [1 / self.config.axle_length, -1 / self.config.axle_length]
        ])

    def apply(self, u: np.ndarray, dt: float):
        drag = -self.config.drag_coeff * self.state.twist
        f = 0.1 * self.G * u + drag

        linear, angular = self.state.twist[0, 0], self.state.twist[1, 0]
        heading = self.state.pose[2, 0]

        self.state.pose += dt * np.asmatrix([[linear * np.cos(heading),
                                              linear * np.sin(heading),
                                              angular]]).T
        self.state.twist += dt * f
