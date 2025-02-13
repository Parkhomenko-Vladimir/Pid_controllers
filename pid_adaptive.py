import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



class pid_adaptive_class():
    def __init__(self, Kp, Kd, Ki, dt, adapt_rate = 0.001):
        self.min = -20
        self.max = 20

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.dt = dt
        self.adapt_rate = adapt_rate

        self.ei = 0
        self.ed = 0
        self.pre_err = 0

    def step(self,  measure, target):
        error = target - measure

        self.Kp += self.adapt_rate * abs(error)
        self.Ki += self.adapt_rate * abs(error) * self.dt
        self.Kd += self.adapt_rate * abs((error - self.pre_err) / self.dt)

        u = self.Kp * error + self.Ki * self.ei + self.Kd * self.ed

        self.ei = self.ei + error * self.dt
        self.ed = (error - self.pre_err) / self.dt
        self.pre_err = error

        return u


class DinamicsModule():
    def __init__(self, x, y, dt):
        self.v = 0
        self.omega = 0
        self.theta = 0
        self.x = x
        self.y = y

        self.dt = dt
        self.speed = 0.3

        self.lenth = 1.0
        self.mass = 10.
        self.inertia = 1.

        self.trust_left_log = []
        self.trust_right_log = []
        self.theta_log = []
        self.v_log = []
        self.time_log = [0]

    def step(self, control_signal):
        left_thrust = np.clip(self.speed - control_signal, 0, 4)
        right_thrust = np.clip(self.speed + control_signal, 0, 4)

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

    theta = np.linspace(0, 2 * np.pi, 26)
    r = 100
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    idx =0

    # target = np.array([100, 100])
    # state = np.array([0, 0])
    target = np.array([x[idx], y[idx]])
    state = np.array([0, 0])

    d_m = DinamicsModule(x=state[0], y=state[1], dt=0.01)
    pid = pid_adaptive_class(Kp = 0.75, Kd = 0.5, Ki = 0.05, dt = 0.01)

    # начальное направление
    d_m.theta = 1.5

    trajectory = []
    # time = []
    k_list = []

    for _ in range(15000):

        target_angle = np.arctan2(target[1] - d_m.y, target[0] - d_m.x)
        current_ungle = d_m.theta

        u = pid.step(target= target_angle, measure = current_ungle)
        d_m.step(control_signal=u)

        # Расчет расстояния до цели
        distance_to_target = np.linalg.norm([d_m.x - target[0], d_m.y - target[1]])

        # Обновление состояния
        state[0] = d_m.x
        state[1] = d_m.y
        trajectory.append((state[0], state[1]))

        # time.append(_ * 0.01)

        # Проверка достижения цели
        if np.linalg.norm([state[0] - target[0], state[1] - target[1]]) < 1.5:
            idx += 1
            idx %= 24
            target = np.array([x[idx], y[idx]])
            print("Цель достигнута!")
            # break
            # print(idx)

    path = np.array(trajectory)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(path[:, 0], path[:, 1], label="Траектория", linestyle='dotted')

    # axes[0].plot(x, y, label="окружность", linestyle='dotted')

    axes[0].scatter(target[0], target[1], color="red")
    axes[0].set_title("Движение катамарана по заданному курсу")
    axes[0].set_xlabel("X (м)")
    axes[0].set_ylabel("Y (м)")
    axes[0].legend()
    axes[0].grid()
    axes[0].axis("equal")

    a = np.array(k_list)


    plt.show()