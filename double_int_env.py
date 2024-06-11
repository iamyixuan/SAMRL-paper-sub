import numpy as np
from traj_gen import get_traj


def sample_gp():
    # sampling from a GP is too slow
    def rbf_kernel(x1, x2, variance=1):
        return np.exp(-1 * ((x1 - x2) ** 2) / (2 * variance))

    def gram_matrix(xs):
        return [[rbf_kernel(x1, x2) for x2 in xs] for x1 in xs]

    xs = np.arange(-1, 1, 2 / 2550.0)
    mean = [0.1 for x in xs]
    gram = gram_matrix(xs)
    ys = np.random.multivariate_normal(mean, gram)

    return ys


def get_demand(seed):
    # get acceleration demand from a 3rd order polynomial
    rd = np.random.RandomState(seed)
    c1, c2 = rd.uniform(-2, 2, 2)

    x = np.linspace(0, 1, 2550)
    c_t = c1 * x[700] ** 2 + c2 * x[700]
    c3 = rd.uniform(-c_t, 2)
    return np.clip(c1 * np.power(x, 3) + c2 * np.power(x, 2) + c3 * x, -1, 1)


def smooth_jump(x, steps):
    ch_pt = np.diff(x)
    ch_idx = np.where(np.abs(ch_pt) > 0)[0]

    idx1 = int(ch_idx - steps)
    idx2 = int(ch_idx + steps)
    x_data = [idx1, idx2]
    y_data = [x[idx1], x[idx2]]

    k = np.diff(y_data) / np.diff(x_data)
    b = x[idx1] - k * idx1
    x_transition = np.linspace(idx1, idx2, steps * 2)
    y = k * x_transition + b
    x[idx1:idx2] = y

    return x


class DoubleIntegratorEnv:
    def __init__(self):
        self.dt = 0.002
        self.sigma = 0.01
        self.max_speed = 1.0
        self.max_torque = 1.0
        self.t_cutoff = 700
        self.horizon = 2550

        self.state_dim = 5
        self.action_dim = 1

    def step(self, action):
        x, v, _, _, _ = self.state
        u = np.clip(action[0], -self.max_torque, self.max_torque)
        v += u * self.dt
        x = x + v * self.dt

        ub = self.ub[self.timestep]
        lb = self.lb[self.timestep]
        rho = self.requested_rho[self.timestep]

        self.state = np.asarray([x, v, rho, lb, ub])

        reward_vec = []
        reward_vec.append(
            float(-((action[0] - self.requested_rho[self.timestep]) ** 2))
        )

        if self.state[1] <= ub:
            reward_vec.append(0.0)
        else:
            reward_vec.append(1.0)

        if self.state[1] >= lb:
            reward_vec.append(0.0)
        else:
            reward_vec.append(1.0)
        self.timestep += 1

        if self.timestep >= self.horizon - 1:
            done = True
        else:
            done = False
        return np.asarray(self.state), np.asarray(reward_vec), done

    def reset(self, seed):
        high = np.array([np.pi, 1])
        self.state = [0, 0, 0, 0, 0]  # x=0, v=0, lb=0, ub=0, request=0
        self.requested_rho = self.get_requested(seed)
        self.timestep = 0
        self.ub, self.lb = self.get_cons(seed)
        return np.asarray(self.state)

    def get_cons(self, seed):
        # get constraints
        # speed lower and upper bound
        rs = np.random.RandomState(seed)
        delta_1 = rs.uniform(0.3, 0.8)
        seg_1 = [delta_1] * self.t_cutoff
        delta_2 = rs.uniform(0.5, 1.2)
        seg_2 = [delta_2] * (self.horizon - self.t_cutoff)
        cons = np.asarray(seg_1 + seg_2)

        # set up the lower bound
        d1 = rs.uniform(-0.5, 0)
        d2 = rs.uniform(-1, 0.5)
        s1 = [d1] * self.t_cutoff
        s2 = [d2] * (self.horizon - self.t_cutoff)
        cons2 = np.asarray(s1 + s2)

        cons = smooth_jump(cons, 100)
        cons2 = smooth_jump(cons2, 100)
        return cons, cons2

    def get_requested(self, seed):
        # self.rho = get_traj(5, seed)
        # self.rho = sample_gp()
        self.rho = get_demand(seed)
        return self.rho

    def cons_bounds(self):
        return np.asarray(self.lb[self.timestep])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = DoubleIntegratorEnv()

    fig, axs = plt.subplots(5, 5)
    ax = axs.flatten()

    def plot(seed):
        state = env.reset(seed)
        states = []

        rho = get_demand(seed)
        rho = np.vstack((rho, np.zeros(rho.shape))).T
        for i in range(rho.shape[0]):
            state_next, _, _ = env.step(rho[i])
            states.append(state_next)
        states = np.asarray(states)

        ub_cons, lb_cons = env.get_cons(seed)
        return states, ub_cons, lb_cons

    for i in range(5 * 5):
        states, ub_cons, lb_cons = plot(i)
        ax[i].plot(states[:, 1])
        ax[i].plot(ub_cons)
        ax[i].plot(lb_cons)
        # ax[1].plot(rho[:, 0])

    plt.show()
