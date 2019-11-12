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

    def generateR(self) -> np.ndarray:
        """
        Generate a R matrix that has large covariance in direction
        perpendicular to measurement, ie. in the x axis of the body frame
        :return: R matrix
        """
        lambda_small = 5e-3
        lambda_big = 1e-2

        d = np.asmatrix([[lambda_big, 0],
                         [0, lambda_small]])

        return d

    def update_beacon(self, distance: float, sensor: int) -> np.ndarray:
        """
        Returns R
        :param distance:
        :param sensor:
        :return:
        """

        import math

        print(f"Starting pose: {self.robot.state.pose.T}")

        # Find the position of the sensor reading.
        angle = self.sensor_angles[sensor]
        global_angle = self.robot.state.pose[2, 0] + self.sensor_angles[sensor]

        print(f"Angle to beacon: {angle}, distance: {distance}")

        reading = distance * np.asmatrix([[np.cos(angle), np.sin(angle)]]).T
        reading_global = self.robot.state.pose[:2, 0] + distance * np.asmatrix(
            [[np.cos(global_angle), np.sin(global_angle)]]).T

        print(f"Sensor beacon position: {reading.T}")

        # First, match the beacon to something in the list.
        beacon_index = None
        beacon_cost = None
        for i, beacon in enumerate(self.beacon_positions):
            cost = np.linalg.norm(beacon - reading_global)
            if beacon_index is None or cost < beacon_cost:
                beacon_cost = cost
                beacon_index = i
        beacon = self.beacon_positions[beacon_index]

        delta_x = beacon[0, 0] - self.robot.state.pose[0, 0]
        delta_y = beacon[1, 0] - self.robot.state.pose[1, 0]

        heading = self.robot.state.pose[2, 0]
        rot = np.asmatrix([[math.cos(heading), -math.sin(heading)],
                           [math.sin(heading), math.cos(heading)]]).T

        print(f"self.robot.state.pose:\n{self.robot.state.pose[:2, :]}")
        print(f"beacon:\n{beacon}")
        print(f"dpos:\n{beacon - self.robot.state.pose[:2, :]}")
        transformed_beacon = rot * (beacon - self.robot.state.pose[:2, :])

        print(f"Transformed beacon location: {transformed_beacon.T}")

        # Next, find the expected measurement of that beacon
        error = reading - transformed_beacon

        print(f"Error: {error.T}")

        actual_distance = np.hypot(delta_x, delta_y)
        print(f"sensor: {distance}, estimate: {actual_distance}")

        H = np.asmatrix([
            [-math.cos(heading), -math.sin(heading), -math.sin(heading) * delta_x + math.cos(heading) * delta_y],
            [math.sin(heading), -math.cos(heading), -math.cos(heading) * delta_x - math.sin(heading) * delta_y]
        ])

        print(f"H:\n{H}")

        R = self.generateR()

        print(f"R:\n{R}")

        K = self.P * H.T * np.linalg.inv(H * self.P * H.T + R)

        print(f"K:\n{K}")

        self.P = self.P - K * H * self.P

        self.robot.state.pose += K * error

        print(f"K * error:\n{K * error}")

        return R
