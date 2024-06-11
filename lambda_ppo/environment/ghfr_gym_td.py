import numpy as np
from sklearn.preprocessing import MinMaxScaler as MMS
from joblib import load
import copy
from collections import deque
import pickle
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
                (
                    self.s_tr_ls,
                    self.a_tr_ls,
                    _,
                    self.x_names,
                    self.a_names,
                    self.s_te_ls,
                    self.a_te_ls,
                ) = self.data_loader(data_path_full)
            else:
                self.model = load(model_path)
                (
                    self.s_tr_ls,
                    self.a_tr_ls,
                    _,
                    self.x_names,
                    self.a_names,
                    self.s_te_ls,
                    self.a_te_ls,
                ) = self.data_loader(data_path)
            self.HX_s_tout_id, self.HX_s_tin_id = 9, 8
        else:
            if skip == 1:
                self.model = load(red_model_path_full)
                (
                    self.s_tr_ls,
                    self.a_tr_ls,
                    _,
                    self.x_names,
                    self.a_names,
                    self.s_te_ls,
                    self.a_te_ls,
                ) = self.data_loader(red_data_path_full)
            else:
                self.model = load(red_model_path)
                (
                    self.s_tr_ls,
                    self.a_tr_ls,
                    _,
                    self.x_names,
                    self.a_names,
                    self.s_te_ls,
                    self.a_te_ls,
                ) = self.data_loader(red_data_path)
            self.HX_s_tout_id, self.HX_s_tin_id = 7, 6

        self.skip = skip
        self.time_independent = time_independent
        self.action_dim = len(self.a_names)
        self.sindy_state_dim = self.s_tr_ls[0].shape[1]  # len(self.x_names)
        self.extra_dim = 4 if rem_time else 3
        self.state_dim = (
            self.sindy_state_dim + self.extra_dim
        ) * hist_len  # because we are adding lb/ub of TE1
        self.timestep = 0
        self.rem_time = rem_time
        self.horizon = len(self.s_tr_ls[0])
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

        # scale the action space
        self.scaler = MinMaxScaler(-1, 1)
        demands = [i[:, 0] for i in self.a_tr_ls]
        self.scaler.fit(demands)  # fit the scaler using the requested rho

    def reset(self, train=True, traj_id=None):

        if train:
            self.timestep = 0
            rem_time = -float(self.timestep - self.horizon / 2) / (self.horizon / 2)
            # first we pick one trajectory
            if traj_id is None:
                self.traj_id = np.random.randint(0, len(self.s_tr_ls))
            else:
                self.traj_id = traj_id
            state_evolution = self.s_tr_ls[self.traj_id]
            control_evolution = self.a_tr_ls[self.traj_id]
            t_int = 0
            init_state = state_evolution[t_int]
            self.requested_rho = control_evolution[:, 0][t_int:]

            self.s0_ub, self.s0_lb = self.get_cons()
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
            state_evolution = self.s_te_ls[self.traj_id]
            control_evolution = self.a_te_ls[self.traj_id]
            t_int = 0
            init_state = state_evolution[t_int]
            self.requested_rho = control_evolution[:, 0][t_int:]
            self.s0_ub, self.s0_lb = self.get_cons()
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

    def step(self, action):
        assert self.requested_rho is not None, "Need to reset environment first!"
        assert np.isfinite(action).all(), "Action is not finite!"
        assert np.isfinite(self.state.sum()).all(), "State is not finite!"

        # inverse transform back to the original scale to get proper rewards
        action = self.scaler.inverse_transform(action)
        if action > 0:
            print("Inverse Transformed Action value is", action)

        # watch it, this is a discrete time model
        curr_state = self.state
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
            reward_vec.append(1.0)

        # second constraint on HX_s_tin >= min_t(HX_s_tin)*0.9
        if self.state[0, self.HX_s_tin_id] >= self.s0_lb[self.timestep]:
            reward_vec.append(0.0)
        else:
            reward_vec.append(1.0)

        if self.timestep >= self.horizon - 1:
            done = True
        else:
            done = False
        self.timestep += 1
        return_state = np.concatenate(list(self.state_buffer), axis=0)
        return np.asarray(return_state), np.asarray(reward_vec), done

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

    def get_cons(self):
        origin = 703.177
        delta_1 = np.random.uniform(-0.8, -1.6)  # -1.8 # -1.65,-1.85
        segment_1 = [origin + delta_1] * self.t_cutoff
        delta_2 = np.random.uniform(-1.0, -1.3)  # -2.05 # -1.85,-2.05
        segment_2 = [origin + delta_2] * (self.horizon - self.t_cutoff)
        cons_lb = segment_1 + segment_2
        cons_lb = (
            self.HX_s_tin_transform(np.asarray(cons_lb)) - self.HX_s_tin_scaled_origin
        )

        delta_3 = np.random.uniform(2.0, 4.0)  # 5.5 # 4.5,5.5
        origin = 742.08
        cons_ub = [origin + delta_3] * self.horizon
        cons_ub = (
            self.HX_s_tout_transform(np.asarray(cons_ub)) - self.HX_s_tout_scaled_origin
        )
        return cons_ub, cons_lb

    def get_cons_old(self, state_evolution):
        # for higher order model
        HX_s_tin = state_evolution[:, self.HX_s_tin_id]
        HX_s_tout = state_evolution[:, self.HX_s_tout_id]
        cons_ub = np.asarray([HX_s_tout.max() * 0.9] * len(HX_s_tout))
        cons_lb = np.asarray([HX_s_tin.min() * 0.9] * len(HX_s_tin))
        return cons_ub, cons_lb

    def cons_bounds(self):
        return np.asarray([self.s0_lb[self.timestep]])


if __name__ == "__main__":
    # env = SINDyGYM(reduced_model=True)
    # # print(len(env.x_names), env.x_names[:20])
    # # print(env.s_tr_ls[0].shape, len(env.s_tr_ls))
    # # print(env.s_te_ls[0].shape, len(env.s_te_ls))
    # print(len(env.a_tr_ls), env.a_tr_ls[10].shape)
    # print(env.a_names, env.sindy_state_dim)
    # # state = env.reset()
    # # next_state, reward, done = env.step(np.asarray([0.1])) # in this test, action dim should be 1
    # # print("hello!")
    # # load dmdc and rg results:
    # # mat = np.load("dmdc_rg_dict_test_exp_0.npy", allow_pickle=True)
    # # loaded_dict = mat.flat[0
    # # keys: dict_keys(['X_dmdc', 'Y_dmdc', 'U_dmdc', 'X_srg', 'Y_srg', 'U_srg'])
    import pandas as pd

    env = SINDyGYM(hist_len=1, skip=5, rem_time=False, time_independent=True)

    init_state = env.reset()
    print("Traj ID is", env.traj_id)
    demand = env.requested_rho
    print(demand.shape)
    data = [init_state]
    for i in range(2250):
        state, _, _ = env.step(np.asarray([demand[i]]))
        data.append(state)

    df = pd.DataFrame(data)
    df.to_csv("~/Desktop/sampleSINDyTraj_savedDemand_ID" + str(env.traj_id) + ".csv")
