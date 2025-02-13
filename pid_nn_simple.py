import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class pid_loss():
    def __init__(self,  dt):
        self.min = -20
        self.max = 20

        self.dt = dt

        self.ei = 0
        self.ed = 0
        self.pre_err = 0

    def forward(self, input, measure, target):
        kp, ki, kd = input
        error = target - measure

        u = kp * error + ki * self.ei + kd * self.ed
        # u = torch.clip(u, a_min=self.min, a_max=self.max)

        self.ei = self.ei + error * self.dt
        self.ed = (error - self.pre_err) / self.dt
        self.pre_err = error

        return T.clamp(u, min=self.min, max=self.max)

class Model_class(nn.Module):
    def __init__(self, input_size, output_size = 3, lr =0.000001) -> None:
        super(Model_class, self).__init__()

        self.inp = nn.Linear(input_size,20)
        self.l1 = nn.Linear(20, 20)
        self.l2 = nn.Linear(20, 20)
        self.out = nn.Linear(20, output_size)

        self.opt = T.optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inp):

        x = F.tanh(self.inp(inp.to(self.device)))
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.sigmoid(self.out(x))

        return x * 10

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class DinamicsModule():
    def __init__(self, x, y, dt):
        self.v = 0
        self.omega = 0
        self.theta = 0
        self.x = x
        self.y = y

        self.dt = dt
        # self.time = time

        self.speed = 0.1
        # self.speed = 100

        self.lenth = 1.0
        self.mass = 10.
        self.inertia = 1.

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

    target = np.array([100,100])
    state = np.array([0, 0])

    pid = pid_loss(0.001)
    model = Model_class(6)
    d_m = DinamicsModule(x=state[0], y=state[1], dt=0.01)

    # начальное направление
    d_m.theta = 1.5
    Kp_prev, Kd_prev, Ki_prev = 0,0,0

    trajectory = []
    time = []
    k_list = []
    i = 0

    while not np.linalg.norm([state[0] - target[0], state[1] - target[1]]) < 1.5:

        target_angle = np.arctan2(target[1] - d_m.y, target[0] - d_m.x)

        model.opt.zero_grad()
        data_in = T.tensor([target_angle % np.pi, d_m.theta % np.pi, target_angle - d_m.theta, Kp_prev, Kd_prev, Ki_prev], dtype=T.float32)

        out = model(data_in)

        u = pid.forward(out,
                        measure=T.tensor(d_m.theta, dtype=T.float32),
                        target=T.tensor(target_angle, dtype=T.float32))

        u.requires_grad_(True)
        u.backward()
        model.opt.step()

        # Расчет расстояния до цели
        distance_to_target = np.linalg.norm([d_m.x - target[0], d_m.y - target[1]])

        d_m.step(control_signal=u.item())

        # Обновление состояния
        state[0] = d_m.x
        state[1] = d_m.y
        trajectory.append((state[0], state[1]))

        i += 1

        time.append(i * 0.01)
        k_list.append(out.cpu().detach().numpy())


    path = np.array(trajectory)
    k_list = np.array(k_list)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(path[:, 0], path[:, 1], label="Траектория", linestyle='dotted')
    axes[0].scatter(target[0], target[1], color="red")
    axes[0].set_title("Движение катамарана по заданному курсу")
    axes[0].set_xlabel("X (м)")
    axes[0].set_ylabel("Y (м)")
    axes[0].legend()
    axes[0].grid()
    axes[0].axis("equal")

    axes[1].set_title("Коэфициенты регулятора")
    axes[1].plot(time[:], k_list[:, ], label=['kp', 'ki', 'kd'])
    axes[1].legend()

    plt.show()