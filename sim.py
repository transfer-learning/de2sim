import pygame
import numpy as np
import random
import vis
import sys

from controller import Controller

from de2bot import DE2Bot, DE2Config


def meas_prob(x, o, meas_angle):
    ddir = o - x[:2, :]
    dist = np.linalg.norm(ddir)
    ddir = ddir / dist

    abs_angle = x[2, 0] + meas_angle
    adir = np.asmatrix([[np.cos(abs_angle), np.sin(abs_angle)]]).T

    cos_th = ddir.T.dot(adir)
    cos = np.arccos(cos_th)
    return np.exp(-35 * dist * cos[0, 0] * cos[0, 0])
    # deg = 30
    # rad = np.radians(deg)
    # return 1 if np.abs(cos) < rad else 0


def sensor_model(x, obstacles, meas_angle):
    by_prox = sorted(obstacles, key=lambda o: np.linalg.norm(x[:2, :] - o))

    for obs in by_prox:
        prob = meas_prob(x, obs, meas_angle)
        if random.random() < prob:
            return np.linalg.norm(obs - x[:2, 0])
    return None


def wait():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                return


def main():
    clock = pygame.time.Clock()

    visualizer = vis.Visualizer()

    fps = 60
    dt = 1.0 / fps
    t = 0.0
    i = 0

    robot = DE2Bot()
    dead_reckon = DE2Bot()

    # obstacles = [
    #     np.asmatrix([[0.5, 0]]).T,
    #     np.asmatrix([[1, 1]]).T,
    #     np.asmatrix([[0.5, 1.5]]).T,
    #     np.asmatrix([[-0.5, 0]]).T,
    # ]
    obstacles = [
        np.asmatrix([[-1.5, -0.5]]).T,
        np.asmatrix([[1.5, -0.5]]).T,
    ]

    angles_deg = [-144, -90, -44, -12, 12, 44, 90, 144]
    angles_rad = [np.deg2rad(a) for a in angles_deg]

    controller = Controller(robot.state.pose, angles_rad, DE2Config(), obstacles)

    next_sensor = None

    framerate = 60.

    # sensor_update_time = 1. / 16.
    sensor_update_time = 1. / 30.
    last_sensor_update = pygame.time.get_ticks()

    hits = []
    while not visualizer.close:
        visualizer.update_events()

        pos = robot.state.pose

        # for a in angles_rad:
        #     dist = sensor_model(pos, obstacles, a)
        #     if dist is not None:
        #         angle = a + pos[2, 0]
        #         hits.append((pos[0, 0] + dist * np.cos(angle), pos[1, 0] + dist * np.sin(angle)))

        sense = None
        if next_sensor:
            a = angles_rad[next_sensor]
            sense = sensor_model(pos, obstacles, a)

            for b in angles_rad:
                if b != a:
                    angle = b + pos[2, 0]
                    hits.append((pos[0, 0] + 0.1 * np.cos(angle), pos[1, 0] + 0.1 * np.sin(angle)))

            if sense is not None:
                angle = a + pos[2, 0]
                hits.append((pos[0, 0] + sense * np.cos(angle), pos[1, 0] + sense * np.sin(angle)))

        can_sense = pygame.time.get_ticks() > (last_sensor_update + sensor_update_time * 1000)
        if can_sense:
            hits = []

        encoder_noise = 0.10 * np.asmatrix(np.random.normal(size=(2, 1)))
        dead_reckon.apply(robot.left_right() + encoder_noise, dt)

        controls, next_sensor, R = controller.update(
            robot.left_right() + encoder_noise,
            sense,
            can_sense,
            1. / framerate
        )
        controls_noise = 0.5 * np.asmatrix(np.random.normal(size=(2, 1))) * np.linalg.norm(controls)
        robot.apply(controls, dt)

        if can_sense and next_sensor:
            last_sensor_update = pygame.time.get_ticks()

        visualizer.draw(
            robot.state.pose,
            dead_reckon.state.pose,
            controller.estimated_position(),
            R,
            hits,
            [(mat[0, 0], mat[1, 0]) for mat in obstacles])

        # wait()
        clock.tick(framerate)


if __name__ == '__main__':
    main()
