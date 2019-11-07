import de2bot
import numpy as np
from localization import Localizer

class Controller:
    def __init__(self, start_pose: np.matrix, sensor_angles, config: de2bot.DE2Config, beacons):
        state = de2bot.State(start_pose)
        self.angles = sensor_angles
        self.prev_controls = np.asmatrix([0, 0.]).T
        self.next_sense = 0
        self.last_sense = None
        self.localization = Localizer(state, config, sensor_angles, beacons)

    def estimated_position(self):
        return self.localization.robot.state.pose

    def update(self, encoders: np.matrix, meas_dist: float, can_sense: bool, delta_time: float):
        """
        Update logic for the controller. Called at 100Hz.

        Parameters:
          encoders: wheel velocities
          meas_dist: previous sonar measurement distance, or None
          can_sense: are we able to sense from sonar
          delta_time: time since last update
        Return:
          (controls, sensor number)
        """
        controls = np.asmatrix([
            [0.2],
            [0.3],
        ])

        # Kalman Filter predict step
        self.localization.predict(encoders, delta_time)

        if meas_dist is not None:
            # Update EKF
            self.localization.update_beacon(meas_dist, self.last_sense)

        self.prev_controls = controls

        sensor = None

        if can_sense:
            sensor = self.next_sense
            self.last_sense = sensor
            self.next_sense = (self.next_sense + 1) % len(self.angles)

        return (controls, sensor)
