import copy
import pickle
import numpy as np
from joblib import load
from collections import deque
from .traj_gen import get_traj
from ..utils.scaler import MinMaxScaler


# pysindy==1.4.3
# scipy==1.6.2
# scikit-learn==0.24.2
# numpy==1.20.3
# sqlitedict==1.7.0
model_path = "./env_data/SINDYc_model_2022-4-15-downsample.joblib"
model_path_full = "./env_data/SINDYc_model_2022-4-15-full.joblib"
data_path = "./env_data/skip5_traj_data.pkl"
data_path_full = "./env_data/full_traj_data.pkl"


red_model_path = "./env_data/red_SINDYc_model_2022-4-15-downsample.joblib"
red_model_path_full = "./env_data/red_SINDYc_model_2022-4-15-full.joblib"
red_data_path = "./env_data/red_skip5_traj_data.pkl"
red_data_path_full = "./env_data/red_full_traj_data.pkl"


def smooth_jump(x, steps, skip=5):
    ch_pt = np.diff(x)
    if skip == 1:
        ch_idx = 700 * 5  # int(np.where(np.abs(ch_pt) > 0)[0])
    else:
        ch_idx = 700
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


class SINDyGYM:
    def __init__(
        self,
        skip=5,
        hist_len=1,
        rem_time=False,
        time_independent=True,
        noisy_obs=False,
        noise_scale=0.01,
        reduced_model=True,
    ):
        if not reduced_model:
            if skip == 1:
                self.model = load(model_path_full)
            else:
                self.model = load(model_path)
            self.HX_s_tout_id, self.HX_s_tin_id = 9, 8
        else:
            if skip == 1:
                self.model = load(red_model_path_full)
            else:
                self.model = load(red_model_path)
            self.HX_s_tout_id, self.HX_s_tin_id = 7, 6

        self.skip = skip
        self.time_independent = time_independent
        self.action_dim = 1  # len(self.a_names)
        self.sindy_state_dim = 13  # self.s_tr_ls[0].shape[1] #len(self.x_names)
        self.extra_dim = 4 if rem_time else 3
        self.reduced_dim = False  # turn it to False with using the full state variables
        self.future_dmd = False
        if self.reduced_dim:
            self.state_dim = 2 + self.extra_dim
        if self.future_dmd:
            self.state_dim = 10 + self.sindy_state_dim + self.extra_dim
        else:
            self.state_dim = (
                self.sindy_state_dim + self.extra_dim
            ) * hist_len  # because we are adding lb/ub of TE1
        self.timestep = 0
        self.rem_time = rem_time
        self.horizon = 11250 // self.skip  # len(self.s_tr_ls[0])
        print('Environment horizon is {} with skip {}'.format(self.horizon, self.skip))
        assert self.horizon == 11250 // self.skip
        self.hist_len = hist_len
        self.state_buffer = deque(maxlen=hist_len)
        self.noisy_obs = noisy_obs
        self.noise_scale = noise_scale
        for _ in range(hist_len):
            self.state_buffer.append(
                np.zeros(shape=(self.sindy_state_dim + self.extra_dim,))
            )

        # constraint transformer
        self.HX_s_tout_transform = (
            lambda x: 0.13931207525540368 * x - 103.27365516795719
        )
        self.HX_s_tin_transform = lambda x: 0.3611462911292418 * x - 253.0496946387899
        if skip == 1:
            self.t_cutoff = 3500  # (700s/0.02 = 3500)
        else:
            # skip == 5
            self.t_cutoff = 700
        self.HX_s_tin_scaled_origin = 0.89999725
        self.HX_s_tout_scaled_origin = 0.10703998

    def change_mode(self, if_reduced_dim=False, if_future_dmd=False):
        self.reduced_dim = if_reduced_dim
        self.future_dmd = if_future_dmd

        if self.reduced_dim:
            self.state_dim = 2 + self.extra_dim
        if self.future_dmd:
            self.state_dim = 8 + self.sindy_state_dim + self.extra_dim
        else:
            pass

        return
    
    def predict(self, action, state):
        next_state = self.model.predict(
            x=state, u=action.reshape(1, -1)
        )  
        return next_state
         

    def reset(self, train=True, traj_id=None, seed=None):

        if train:
            self.timestep = 0
            rem_time = -float(self.timestep - self.horizon / 2) / (self.horizon / 2)
            init_state = np.zeros(13)  # 13 initial state variables
            """
            requeted_rho:
                the demand randomly generated from the new traj_gen class
            """
            self.requested_rho = get_traj(self.skip, seed)

            self.s0_ub, self.s0_lb = self.get_cons(seed)
            self.state = init_state.reshape(1, -1)
            self.state_buffer = deque(maxlen=self.hist_len)
            for _ in range(self.hist_len):
                self.state_buffer.append(
                    np.zeros(shape=(self.sindy_state_dim + self.extra_dim,))
                )
            agent_state = copy.deepcopy(self.state)
            if self.noisy_obs:
                agent_state += np.random.normal(
                    loc=0, scale=self.noise_scale, size=(1, self.sindy_state_dim)
                )
            if self.rem_time:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                        + [rem_time]
                    )
                )
            elif self.reduced_dim:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel()[[self.HX_s_tout_id, self.HX_s_tin_id]])
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                    )
                )  # only input the contrainted inputs
            elif self.future_dmd:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + list(self.s0_lb[self.timestep : self.timestep + 5])
                        + list(self.s0_ub[self.timestep : self.timestep + 5])
                    )
                )  # the number here is the future steps of demand to input.
            else:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                    )
                )

            return_state = np.concatenate(list(self.state_buffer), axis=0)
            return np.asarray(return_state)
        else:
            self.timestep = 0
            rem_time = -float(self.timestep - self.horizon / 2) / (self.horizon / 2)
            self.traj_id = traj_id
            self.requested_rho = get_traj(self.skip)
            self.s0_ub, self.s0_lb = self.get_cons()
            init_state = np.zeros(13)
            self.state = init_state.reshape(1, -1)
            self.state_buffer = deque(maxlen=self.hist_len)
            for _ in range(self.hist_len):
                self.state_buffer.append(
                    np.zeros(shape=(self.sindy_state_dim + self.extra_dim,))
                )
            agent_state = np.copy(self.state)  # copy.deepcopy(self.state)
            if self.noisy_obs:
                agent_state += np.random.normal(
                    loc=0, scale=self.noise_scale, size=(1, self.sindy_state_dim)
                )
            if self.rem_time:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                        + [rem_time]
                    )
                )
            elif self.reduced_dim:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel()[[self.HX_s_tout_id, self.HX_s_tin_id]])
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                    )
                )  # only input the contrainted inputs
            elif self.future_dmd:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + list(self.s0_lb[self.timestep : self.timestep + 5])
                        + list(self.s0_ub[self.timestep : self.timestep + 5])
                    )
                )  # the number here is the future steps of demand to input.
            else:
                self.state_buffer.append(
                    np.asarray(
                        list(agent_state.ravel())
                        + [0.0]
                        + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                    )
                )
            return_state = np.concatenate(list(self.state_buffer), axis=0)
            return np.asarray(return_state)

    def step(self, action, PID):
        assert self.requested_rho is not None, "Need to reset environment first!"
        # assert np.isfinite(action).all(), "Action is not finite!"
        assert np.isfinite(self.state.sum()).all(), "State is not finite!"


        # if the action is not finite, ignore episode
        curr_state = self.state
        if not np.isfinite(action).all():
            f = True
            next_state = self.state
        else:
        # watch it, this is a discrete time model
            next_state = self.model.predict(
                x=curr_state, u=action.reshape(1, -1)
            )  # this needs the original scaled action
        # to avoid the agent drawing the model outside of correct range.
        clipped_next_state = np.clip(next_state, -2.0, 2.0)
        done = False
        # this is to terminate the episode if sindy model is driven outside training data bounds
        rem_time = -float((self.timestep + 1) - self.horizon / 2) / (self.horizon / 2)
        self.state = clipped_next_state
        agent_state = copy.deepcopy(self.state)
        if self.noisy_obs:
            agent_state += np.random.normal(
                loc=0, scale=self.noise_scale, size=(1, self.sindy_state_dim)
            )
        if self.rem_time:
            self.state_buffer.append(
                np.asarray(
                    list(agent_state.ravel())
                    + [self.requested_rho[self.timestep]]
                    + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                    + [rem_time]
                )
            )
        elif self.reduced_dim:
            self.state_buffer.append(
                np.asarray(
                    list(agent_state.ravel()[[self.HX_s_tout_id, self.HX_s_tin_id]])
                    + [self.requested_rho[self.timestep]]
                    + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                )
            )  # only input the contrainted inputs
        elif self.future_dmd:
            self.state_buffer.append(
                np.asarray(
                    list(agent_state.ravel())
                    + [self.requested_rho[self.timestep : self.timestep + 10]]
                    + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                )
            )  # the number here is the future steps of demand to input.
        else:
            self.state_buffer.append(
                np.asarray(
                    list(agent_state.ravel())
                    + [self.requested_rho[self.timestep]]
                    + [self.s0_lb[self.timestep], self.s0_ub[self.timestep]]
                )
            )

        reward_vec = []
        reward_vec.append(
            -((action[0] - self.requested_rho[self.timestep]) ** 2)
        )  # primary objective
        # first constraint HX_s_tout <= max_t(HX_s_tout)*0.9
        if self.state[0, self.HX_s_tout_id] <= self.s0_ub[self.timestep]:
            reward_vec.append(0.0)
        else:
            reward_vec.append(1.0) # change to the magnitude of violation or the squared distance.
            # reward_vec.append(
            #     (self.state[0, self.HX_s_tout_id] - self.s0_ub[self.timestep]) ** 2
            # )

        # second constraint on HX_s_tin >= min_t(HX_s_tin)*0.9
        if self.state[0, self.HX_s_tin_id] >= self.s0_lb[self.timestep]:
            reward_vec.append(0.0)
        else:
            reward_vec.append(1.0)
            # reward_vec.append(
            #     (self.state[0, self.HX_s_tin_id] - self.s0_lb[self.timestep]) ** 2
            # )

        if self.timestep >= self.horizon - 1:
            done = True
        else:
            done = False
        self.timestep += 1
        return_state = np.concatenate(list(self.state_buffer), axis=0)

        if PID == 5:
            f = True
        else:
            f = False
        return np.asarray(return_state), np.asarray(reward_vec), done, f

    def data_loader(self, file_path):
        with open(file_path, "rb") as file:
            data_dict = pickle.load(file)
        s_tr_ls = data_dict["s_tr_ls"]
        a_tr_ls = data_dict["a_tr_ls"]
        t_tr_ls = data_dict["t_tr_ls"]
        X_Names = data_dict["X_Names"]
        U_Names = data_dict["U_Names"]
        s_te_ls = data_dict["s_te_ls"]
        a_te_ls = data_dict["a_te_ls"]
        return s_tr_ls, a_tr_ls, t_tr_ls, X_Names, U_Names, s_te_ls, a_te_ls

    def get_cons(self, seed):
        rs = np.random.RandomState(seed)
        origin = 703.177
        delta_1 = rs.uniform(-0.8, -1.6)  # -1.8 # -1.65,-1.85
        segment_1 = [origin + delta_1] * self.t_cutoff
        delta_2 = rs.uniform(-1.0, -1.3)  # -2.05 # -1.85,-2.05 # (-1, -1.3)
        segment_2 = [origin + delta_2] * (self.horizon - self.t_cutoff)
        cons_lb = segment_1 + segment_2
        cons_lb = (
            self.HX_s_tin_transform(np.asarray(cons_lb)) - self.HX_s_tin_scaled_origin
        )

        delta_3 = rs.uniform(2.0, 4.0)  # 5.5 # 4.5,5.5
        origin = 742.08  # lower the upper bounds originally 742.08
        cons_ub = [origin + delta_3] * self.horizon
        cons_ub = (
            self.HX_s_tout_transform(np.asarray(cons_ub)) - self.HX_s_tout_scaled_origin
        )

        cons_lb = smooth_jump(np.asarray(cons_lb), int(100 * 5 / self.skip), self.skip)
        cons_ub = smooth_jump(np.asarray(cons_ub), int(100 * 5 / self.skip), self.skip)
        return cons_ub, cons_lb

    def cons_bounds(self):
        return np.asarray([self.s0_lb[self.timestep]])


if __name__ == "__main__":
    import pandas as pd

    env = SINDyGYM(hist_len=1, skip=5, rem_time=False, time_independent=True)
    init_state = env.reset()
    demand = env.requested_rho
    print(demand.shape)
    data = [init_state]
    for i in range(2250):
        state, _, _ = env.step(np.asarray([demand[i]]))
        data.append(state)

    df = pd.DataFrame(data)
    df.to_csv("~/Desktop/sampleSINDyTraj.csv")

    # env.change_mode(if_future_dmd=True)
    # print(env.state_dim)
    # o = env.reset(seed=1)
    # print(env.state.shape)
    # o2 = env.step(action=np.ones(1))
    # print(o2[0].shape)
    # print(env.state.shape)
