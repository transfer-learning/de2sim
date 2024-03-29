import pygame
import numpy as np
import math


class Visualizer:
    def __init__(self):
        self.resolution = (800, 600)
        self.display = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("TL-DE2SIM")
        self.close = False
        self.keys = set([])
        self.ppm = 200
        self.radius = 0.09

    def update_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close = True
            elif event.type == pygame.KEYDOWN:
                self.keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys.remove(event.key)

    def extract_wheel_inputs(self):
        u = np.asmatrix(np.zeros((4, 1)))
        max_torque = 0.780
        if ord('1') in self.keys:
            u[0, 0] = max_torque
        if ord('2') in self.keys:
            u[0, 0] = -max_torque
        if ord('3') in self.keys:
            u[1, 0] = max_torque
        if ord('4') in self.keys:
            u[1, 0] = -max_torque
        if ord('q') in self.keys:
            u[2, 0] = max_torque
        if ord('w') in self.keys:
            u[2, 0] = -max_torque
        if ord('e') in self.keys:
            u[3, 0] = max_torque
        if ord('r') in self.keys:
            u[3, 0] = -max_torque
        return u

    def extract_goals(self):
        r = np.asmatrix(np.zeros((3, 1)))
        speed_max = 8
        arate_max = 8
        if ord('w') in self.keys:
            r[1, 0] = speed_max
        if ord('s') in self.keys:
            r[1, 0] = -speed_max
        if ord('a') in self.keys:
            r[0, 0] = -speed_max
        if ord('d') in self.keys:
            r[0, 0] = speed_max
        if ord('q') in self.keys:
            r[2, 0] = arate_max
        if ord('e') in self.keys:
            r[2, 0] = -arate_max
        return r

    def draw(self, robot_coords, dr_coords, estimation, R, ping_coords, obs_coords):
        """
        Visualize things
        :param robot_coords:
        :param dr_coords:
        :param estimation:
        :param R: R matrix to draw ellipse
        :param ping_coords:
        :param obs_coords:
        :return:
        """
        screen_coords = self.to_screen_coords(robot_coords)

        heading = robot_coords[2]
        line_end_coords = (
            screen_coords[0] + int(self.ppm * self.radius * 4 * np.cos(heading)),
            screen_coords[1] - int(self.ppm * self.radius * 4 * np.sin(heading))
        )
        radius = int(self.ppm * self.radius)

        self.display.fill((0, 0, 0))
        pygame.draw.circle(self.display, (255, 0, 0), screen_coords, radius)
        pygame.draw.aaline(self.display, (255, 0, 0),
                           screen_coords, line_end_coords, 10)

        # Estimate
        rscreen_coords = self.to_screen_coords(estimation)
        rline_end_coords = (
            rscreen_coords[0] + int(self.ppm * self.radius * 4 * np.cos(estimation[2])),
            rscreen_coords[1] - int(self.ppm * self.radius * 4 * np.sin(estimation[2]))
        )
        pygame.draw.aaline(self.display, (0, 255, 0),
                           rscreen_coords, rline_end_coords, 10)
        pygame.draw.circle(self.display, (0, 255, 0),
                           rscreen_coords, 10)

        # Dead Reckoning
        dr_screen_coords = self.to_screen_coords(dr_coords)
        dr_line_end_coords = (
            dr_screen_coords[0] + int(self.ppm * self.radius * 4 * np.cos(dr_coords[2])),
            dr_screen_coords[1] - int(self.ppm * self.radius * 4 * np.sin(dr_coords[2]))
        )
        pygame.draw.aaline(self.display, (0, 0, 255),
                           dr_screen_coords, dr_line_end_coords, 10)
        pygame.draw.circle(self.display, (0, 0, 255),
                           dr_screen_coords, 10)

        # Covariance ellipse
        # if R is not None:
        #     lambda_, v = np.linalg.eig(R)
        #     lambda_ = np.sqrt(lambda_)
        #
        #     F = np.asmatrix([[lambda_[0] * 2, lambda_[1] * 2]]).T
        #
        #     ang = np.arccos(v[0, 0])
        #     F = np.asmatrix([[math.cos(ang), -math.sin(ang)],
        #                      [math.sin(ang), math.cos(ang)]]) * F
        #
        #     ell_radius_x = np.abs(F[0, 0])
        #     ell_radius_y = np.abs(F[1, 0])
        #
        #     print(f"x: {ell_radius_x}, y: {ell_radius_y}")
        #
        #     obs_0 = obs_coords[0]
        #     bl = self.to_screen_coords((obs_0[0] - (ell_radius_x / 2), obs_0[1] + (ell_radius_y / 2)))
        #
        #     rect = pygame.rect.Rect(bl[0], bl[1], ell_radius_x * self.ppm, ell_radius_y * self.ppm)
        #     pygame.draw.ellipse(self.display, (255, 0, 0), rect, 1)
        #     pygame.draw.aaline(self.display, (255, 255, 255), self.to_screen_coords((obs_0[0])))

        for ping in ping_coords:
            ping_pos = self.to_screen_coords(ping)
            robot_screen_coords = self.to_screen_coords(robot_coords[:, 0])

            dx = ping[0] - robot_coords[0, 0]
            dy = ping[1] - robot_coords[1, 0]

            line_angle = math.atan2(dy, dx)

            dy = ping_pos[0] - robot_screen_coords[0]
            dx = ping_pos[1] - robot_screen_coords[1]

            line_length = math.hypot(dx, dy)

            start_angle = line_angle - 10 * np.pi / 180
            end_angle = line_angle + 10 * np.pi / 180

            rect = pygame.Rect(robot_screen_coords[0] - line_length, robot_screen_coords[1] - line_length,
                               line_length * 2, line_length * 2)
            # pygame.draw.rect(self.display, (255, 255, 255), rect, 1)
            pygame.draw.arc(self.display, (255, 255, 255), rect, start_angle, end_angle)
            pygame.draw.aaline(self.display, (255, 255, 255),
                               screen_coords, self.to_screen_coords(ping), 10)

        for obs in obs_coords:
            pygame.draw.circle(self.display, (0, 0, 255),
                               self.to_screen_coords(obs), 10)

        pygame.display.update()

    def to_screen_coords(self, pos):
        return (int(pos[0] * self.ppm + self.resolution[0] / 2),
                int(self.resolution[1] / 2 - pos[1] * self.ppm))
