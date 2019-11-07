import de2bot as de2
import numpy as np

class Localizer:
    def __init__(self, initial_state, config, sensors, beacon_positions):
        self.beacon_positions = beacon_positions
        self.sensor_angles = sensors
        self.robot = de2.DE2Bot(initial_state, config)
        self.P = np.asmatrix([
            [0.05, 0, 0.],
            [0., 0.05, 0.],
            [0., 0., 0.05],
        ])
        self.A = np.asmatrix(np.eye(3, 3))
        self.Q = np.asmatrix(0.1 * np.eye(3, 3))
        self.R = np.asmatrix(0.05 * np.eye(2, 2))

    def predict(self, wheel_velocities: np.matrix, dt: float):
        self.robot.apply(wheel_velocities, dt)
        Pdot = 2 * self.P + self.Q
        self.P = Pdot * dt

    def update_beacon(self, distance: float, sensor: int):
        # Find the position of the sensor reading.
        angle = self.sensor_angles[sensor] + self.robot.state.pose[2, 0]
        reading = distance * np.asmatrix([[np.cos(angle), np.sin(angle)]]).T

        # First, match the beacon to something in the list.
        beacon_index = None
        beacon_cost = None
        for i, beacon in enumerate(self.beacon_positions):
            cost = np.linalg.norm(beacon - reading)
            if beacon_index is None or cost < beacon_cost:
                beacon_cost = cost
                beacon_index = i
        beacon = self.beacon_positions[beacon_index]

        # Next, find the expected measurement of that beacon
        error = reading - beacon

        H = np.asmatrix([
            [-1, 0, 0],
            [0, -1, 0]
        ])
        K = self.P * H.T * np.linalg.inv(H * self.P * H.T + self.R)
        self.P = self.P - K * H * self.P
        self.robot.state.pose += K * error
        print(K * error)
