import pygame
import numpy as np
import random
import vis

from de2bot import DE2Bot


def meas_prob(x, o, meas_angle):
    ddir = o - x[:2, :]
    dist = np.linalg.norm(ddir)
    ddir = ddir / dist

    abs_angle = x[2, 0] + meas_angle
    adir = np.asmatrix([[np.cos(abs_angle), np.sin(abs_angle)]]).T

    cos_th = ddir.T.dot(adir)
    cos = np.arccos(cos_th)
    return np.exp(-15 * dist * cos[0, 0] * cos[0, 0])


def sensor_model(x, obstacles, meas_angle):
    by_prox = sorted(obstacles, key=lambda o: np.linalg.norm(x[:2, :] - o))

    for obs in by_prox:
        prob = meas_prob(x, obs, meas_angle)
        if random.random() < prob:
            return np.linalg.norm(obs - x[:2, 0])
    return None


def main():
    clock = pygame.time.Clock()

    visualizer = vis.Visualizer()

    fps = 60
    dt = 1.0 / fps
    t = 0.0
    i = 0

    robot = DE2Bot()

    obstacles = [
        np.asmatrix([[0.5, 0]]).T,
        np.asmatrix([[1, 1]]).T,
        np.asmatrix([[0.5, 1.5]]).T,
        np.asmatrix([[-0.5, 0]]).T,
    ]

    while not visualizer.close:
        visualizer.update_events()

        angles_deg = [-144, -90, -44, -12, 12, 44, 90, 144]
        angles_rad = [np.deg2rad(a) for a in angles_deg]

        pos = robot.state.pose

        hits = []
        for a in angles_rad:
            dist = sensor_model(pos, obstacles, a)
            if dist is not None:
                angle = a + pos[2, 0]
                hits.append((pos[0, 0] + dist * np.cos(angle), pos[1, 0] + dist * np.sin(angle)))

        u = np.asmatrix([1.0, 2.0]).T
        robot.apply(u, dt)

        visualizer.draw(robot.state.pose, robot.state.pose, hits, [(mat[0, 0], mat[1, 0]) for mat in obstacles])

        clock.tick(60)


if __name__ == '__main__':
    main()
