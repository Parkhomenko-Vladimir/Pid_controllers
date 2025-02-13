import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class pid_simple():
    def __init__(self,  dt = 0.01):
        self.min = -20
        self.max = 20

        self.kp = 1.9
        self.ki = 0.4
        self.kd = 0.6

        self.dt = dt

        self.ei = 0
        self.ed = 0

        self.pre_err = 0

    def step(self, measure, target):
        error = target - measure

        u = self.kp * error + self.ki * self.ei + self.kd * self.ed

        self.ei = self.ei + error * self.dt
        self.ed = (error - self.pre_err) / self.dt
        self.pre_err = error

        return np.clip(u, a_min=self.min, a_max=self.max)

class DinamicsModule():
    def __init__(self, x, y, dt):
        self.v = 0
        self.omega = 0
        self.theta = 0
        self.x = x
        self.y = y

        self.dt = dt
        self.speed = 0.1

        self.lenth = 1.0
        self.mass = 10.
        self.inertia = 0.1

        self.trust_left_log = []
        self.trust_right_log = []
        self.theta_log = []
        self.v_log = []
        self.time_log = [0]

    def step(self, control_signal):
        left_thrust = np.clip(self.speed - control_signal, 0, 100)
        right_thrust = np.clip(self.speed + control_signal, 0, 100)

        force = left_thrust + right_thrust
        torque = (right_thrust - left_thrust) * self.lenth / 2.0

        acceleration = force / self.mass
        angular_acceleration = torque / self.inertia

        self.v += acceleration * self.dt
        self.omega += angular_acceleration * self.dt
        self.theta += self.omega * self.dt

        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        self.trust_left_log.append(left_thrust)
        self.trust_right_log.append(right_thrust)
        self.theta_log.append(self.theta)
        self.v_log.append(self.v)
        self.time_log.append(self.time_log[-1] + self.dt)


if __name__ == '__main__':
    # целевое и начальное положения в пространстве


    target = np.array([100, 70])
    state = np.array([0, 0])

    d_m = DinamicsModule(x=state[0], y=state[1], dt=0.01)
    pid = pid_simple()

    # начальное направление
    d_m.theta = 1.5

    trajectory = []
    # time = []
    k_list = []

    while not np.linalg.norm([state[0] - target[0], state[1] - target[1]]) < 1.5:

        target_angle = np.arctan2(target[1] - d_m.y, target[0] - d_m.x)
        current_ungle = d_m.theta

        u = pid.step(target = target_angle, measure = current_ungle)
        d_m.step(control_signal=u)

        # Расчет расстояния до цели
        distance_to_target = np.linalg.norm([d_m.x - target[0], d_m.y - target[1]])

        # Обновление состояния
        state[0] = d_m.x
        state[1] = d_m.y
        trajectory.append((state[0], state[1]))

    path = np.array(trajectory)

    plt.plot(path[:, 0], path[:, 1], label="Траектория", linestyle='dotted')

    plt.scatter(target[0], target[1], color="red")
    plt.suptitle("Движение катамарана по заданному курсу")
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    plt.show()