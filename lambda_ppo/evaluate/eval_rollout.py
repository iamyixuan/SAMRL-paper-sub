import pickle
import os

import tensorflow as tf
import pandas as pd
import numpy as np

from tqdm import tqdm
from ..model.lstm_actor import Actor  # this import should be changed
from ..model.actor import Actor as Actor_MLP
from joblib import load
from ..environment.ghfr_gym_rd import SINDyGYM


def distance(states, constraints):
    """
    Two-element tuple for agent's performance evaluation
    - The first element being none-zero if any violation happened, and the distance from the state to its constraint.
    - The second element being none-zero if no violation happened and the distance from the state to its constraint
    args:
        states: shape [trajLength, 2], constaining the RL agent controlled inlet/outlet water temperature.
        constraints: shape [trajLength, 2], constraining the safety contraints for inlet/outlet water temperature.
    """
    d_in = np.zeros((states.shape[0], 2))
    d_out = np.zeros((states.shape[0], 2))
    s_in = states[:, 0]
    s_out = states[:, 1]
    c_in = constraints[:, 0]
    c_out = constraints[:, 1]

    # calculate for the inlet: positive number -> no violation; negative number -> violation
    diff_in = s_in - c_in
    v_mask_in = diff_in < 0
    d_in[v_mask_in, 0] = np.abs(diff_in[v_mask_in])
    d_in[~v_mask_in, 1] = np.abs(diff_in[~v_mask_in])

    # calculate for the outlet: positive number -> no violation; negative number -> violation
    diff_out = c_out - s_out
    v_mask_in_out = diff_out < 0
    d_out[v_mask_in_out, 0] = np.abs(diff_out[v_mask_in_out])
    d_out[~v_mask_in_out, 1] = np.abs(diff_in[~v_mask_in_out])
    d = np.mean((d_in + d_out) / 2, axis=0)
    return d


def violation_proportion(states, constraints):
    """
    Calucate the proportion of violation of a sinlge trajectory
    args:
        states: shape [trajLength, 2], constaining the RL agent controlled inlet/outlet water temperature.
        constraints: shape [trajLength, 2], constraining the safety contraints for inlet/outlet water temperature.
    """

    v_in = np.zeros(states.shape[0])
    v_out = np.zeros(states.shape[0])
    s_in = states[:, 0]
    s_out = states[:, 1]
    c_in = constraints[:, 0]
    c_out = constraints[:, 1]

    # calculate for the inlet: positive number -> no violation; negative number -> violation
    diff_in = s_in - c_in
    v_mask_in = diff_in < 0
    v_in[v_mask_in] = 1

    # calculate for the outlet: positive number -> no violation; negative number -> violation
    diff_out = c_out - s_out
    v_mask_in_out = diff_out < 0
    v_out[v_mask_in_out] = 1

    p_in = np.sum(v_in) / states.shape[0]
    p_out = np.sum(v_out) / states.shape[0]
    v_proportion = (p_in + p_out) / 2
    return v_proportion


class AgentEval:
    def __init__(
        self,
        # sindy_path,
        policy_model_weigths,
        policy_config,
        seed,
        env_steps,
        lstm_actor="True",
        if_red="False",
        if_interp="False"
    ):
        # self.sindy_path = sindy_path
        self.policy_model_weights = policy_model_weigths

        self.lstm_actor = lstm_actor
        self.if_red = if_red
        self.policy_config = policy_config
        self.seed = seed

        # constraint and state inverse transformers
        self.HX_s_tout_in_tran = (
            lambda x: (x + 103.27365516795719) / 0.13931207525540368
        )
        self.HX_s_tin_in_tran = lambda x: (x + 253.0496946387899) / 0.3611462911292418

        self.HX_s_tin_scaled_origin = 0.89999725
        self.HX_s_tout_scaled_origin = 0.10703998

        if env_steps == 2250:
            self.SKIP = 5
            self.STEPS = 2250
        elif env_steps == 11250:
            self.SKIP = 1
            self.STEPS = 11250
        
        if if_interp == "True":
            self.if_interp = True
        else:
            self.if_interp = False

        self.skip5_env = SINDyGYM(hist_len=1, skip=5, rem_time=False, time_independent=True)

        self.policy_model = self.load_policy(
            action_dim=policy_config["action_dim"],
            obs_dim=policy_config["obs_dim"],
            pi_layers=policy_config["pi_layers"],
            activation=policy_config["activation"],
            output_activation=policy_config["output_activation"],
            control_type=policy_config["control_type"],
            h_size=policy_config["h_size"],
            batch_size=policy_config["batch_size"],
            safety_layer=policy_config["safety_layer"],
        )

        # import pandas as pd
        # df = pd.read_csv('~/Downloads/test.csv')
        # req = df['RL'].values
        # self.req = (req - 320) / 128.

    def load_policy(
        self,
        action_dim,
        obs_dim,
        pi_layers,
        activation,
        output_activation,
        control_type,
        h_size,
        batch_size,
        safety_layer,
    ):
        with open(self.policy_model_weights, "rb") as file:
            policy_weights = pickle.load(file)

        if self.lstm_actor == "True":
            policy_model = Actor(
                action_dim=action_dim,
                hidden_sizes=pi_layers,
                activation=activation,
                output_activation=output_activation,
                control_type=control_type,
                h_size=h_size,
                batch_size=batch_size,
                safety_layer=safety_layer,
            )
            trace_len = self.STEPS // batch_size
            print("Trace lenght is", trace_len)
            policy_model.build(input_shape=[None, trace_len, obs_dim])
        else:
            policy_model = Actor_MLP(
                action_dim=action_dim,
                hidden_sizes=pi_layers,
                activation=activation,
                output_activation=output_activation,
                control_type=control_type,
                safety_layer=safety_layer,
            )
            policy_model.build(input_shape=[None, obs_dim])

        policy_model.set_weights(policy_weights["policy"])
        return policy_model
    
    def state_pred(self, action, state):
        """
        This method is to use SINDY model (skip=5) to make state predictions for every 5 steps
        in order to get estimated actions from the trained RL model for action interpolation.

        We also need to track the state of the env for rollout to make proper predictions using
        the prediction env.
        """
        next_state = self.skip5_env.predict(action, state) 
        clipped_next_state = np.clip(next_state, -2.0, 2.0)
        return clipped_next_state

    def RL_act(self, o, piLSTM_state, t):
        o = np.asarray(o, dtype=np.float32).reshape(
            1, -1
        )  # repeat unitl a stabilize then update o.
        if self.lstm_actor == "True":
            if t == 0: # burn in the inital hidden state for LSTM actor
                for j in range(128):
                    a, a_std, _, piLSTM_state = self.policy_model.policy_act(
                        obs_ph=o.reshape(1, 1, -1),
                        lstm_state=piLSTM_state,
                        deterministic=True,
                    )

            else:
                a, a_std, _, piLSTM_state = self.policy_model.policy_act(
                    obs_ph=o.reshape(1, 1, -1),
                    lstm_state=piLSTM_state,
                    deterministic=True,
                )
        else:
            a, a_std, _ = self.policy_model.policy_act(
                obs_ph=o.reshape(1, -1), deterministic=True
            )  # MLP actor
        action = a[0].numpy()
        action_std = a_std[0].numpy()
        return action, piLSTM_state

    def demand_follow_state(self):
        env = SINDyGYM(hist_len=1, skip=self.SKIP, rem_time=False, time_independent=True)

        if self.if_red == "True":
            print("Using reduced dimension")
            env.change_mode(if_reduced_dim=True)

        o = env.reset(seed=self.seed)
        states = []

    
        # Output the states exactly following the demands.
        for t in range(self.STEPS):
            action = env.requested_rho[t]
            action = np.asarray([action])
            o2, rew_vec, d = env.step(action)
            states.append(o2)
        states = np.squeeze(states)
        return states

    def get_state_action(self, policy_config):
        # model = load(self.sindy_path)

        states = []
        actions = []
        action_stds = []
        rewards = []

        env = SINDyGYM(hist_len=1, skip=self.SKIP, rem_time=False, time_independent=True)
        # env = DoubleIntegratorEnv()
        o = env.reset(seed=self.seed)
        # cons_bounds=np.asarray(env.cons_bounds()).reshape(1,-1)
        # execute the policy
        piLSTM_state = (
            tf.zeros((1, policy_config["h_size"])),
            tf.zeros((1, policy_config["h_size"])),
        )
        i = 0 
        for t in tqdm(range(self.STEPS)):
        # The hidden state from the current state is part of the next state as input for next action
        # Since SINDY skip=5 predicts the next state (5 steps later) which aligns with the RL model, the hidden state for the predicted
        # state is obtained from the current state
            if self.if_interp:
                print("PERFORMING ACTION INTERPOLATION...")
                if t % 5 == 0:
                    i += 1
                    action, piLSTM_state = self.RL_act(o, piLSTM_state, t)
                # if self.if_interp and t % 5 == 0:
                    # Get the state prediction from the SINDY skip=5 model
                    curr_state = env.state
                    state_p = self.state_pred(action=action, state=curr_state) 
                    # Need to append the demand and bounds for RL actions.
                    try:
                        state_p = np.asarray(list(state_p.ravel()) + [env.requested_rho[t+5]] + [env.s0_lb[t+5], env.s0_ub[t+5]])
                    except:
                        print('Reaching the end at t={}...'.format(t))
                        state_p = np.asarray(list(state_p.ravel()) + [env.requested_rho[t]] + [env.s0_lb[t], env.s0_ub[t]])
                    act_p, _ = self.RL_act(state_p, piLSTM_state, t=1)
                    act_interp = np.interp([1,2,3,4,5], [0, 5], [action[0], act_p[0]]) # the action from RL model at s_t should be dependent on s_{t-5}
                    action = act_interp # the interpolated action is the action for the next state in rollout env!!!
                # if t%5 == 0:
                    for i in range(len(action)):
                        act = np.array([action[i]])
                        o2, rew_vec, d = env.step(act)
                        # x0 = model.predict(x=o.reshape(1,-1), u=action.reshape(1,-1))
                        states.append(o2)
                        actions.append(act)
                        # action_stds.append(action_std)
                        rewards.append(rew_vec) 
                    o = o2
            else:
                action, piLSTM_state = self.RL_act(o, piLSTM_state, t)
                o2, rew_vec, d = env.step(action)
                states.append(o2)
                actions.append(action)
                # action_stds.append(action_std)
                rewards.append(rew_vec) 
                o = o2
        actions = np.asarray(actions)
        states = np.asarray(states)
        rewards = np.asarray(rewards)
        states = np.squeeze(states)
        return states, actions, env.requested_rho[:self.STEPS], rewards

    def rollout_trajectory(self, dir_path, file_name):
        """
        Generate trajectories and save to a csv file
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        states, actions, requested_rho, rewards = self.get_state_action(
            self.policy_config
        )  # add the demand following state as well.
        states_demand = self.demand_follow_state()   
        # Get time steps.
        ts = np.arange(0, len(states[:, 3]))
        # Inversely transform the states and bounds into their original scales and convert from K to C.
        in_temp = (
            self.HX_s_tin_in_tran(states[:, 6]) + self.HX_s_tin_scaled_origin - 273.15
        )
        out_temp = (
            self.HX_s_tout_in_tran(states[:, 7]) + self.HX_s_tout_scaled_origin - 273.15
        )
        lb_cons = (
            self.HX_s_tin_in_tran(states[:, -2]) + self.HX_s_tin_scaled_origin - 273.15
        )
        ub_cons = (
            self.HX_s_tout_in_tran(states[:, -1])
            + self.HX_s_tout_scaled_origin
            - 273.15
        )

        # Get states for calculating the metrics
        s_ = np.vstack([in_temp, out_temp]).T
        # Get constraints for calculating the metrics
        c_ = np.vstack([lb_cons, ub_cons]).T

        d = distance(s_, c_)
        v = violation_proportion(s_, c_)

        # Get unconstrained states
        uc_in_temp = (
            self.HX_s_tin_in_tran(states_demand[:, 6])
            + self.HX_s_tin_scaled_origin
            - 273.15
        )
        uc_out_temp = (
            self.HX_s_tout_in_tran(states_demand[:, 7])
            + self.HX_s_tout_scaled_origin
            - 273.15
        )

        # rescale to the original unit
        actions = actions * 128 + 320
        requested_rho = requested_rho * 128 + 320

        df = {
            "core_flow": states[:, 0],
            "secondary_flow": states[:, 1],
            "pipe110-inT": states[:, 2],
            "core_out_T": states[:, 3],
            "core_in_P": states[:, 4],
            "core_out_P": states[:, 5],
            "core_flow_d": states_demand[:, 0],
            "secondary_flow_d": states_demand[:, 1],
            "pipe110-inT_d": states_demand[:, 2],
            "core_out_T_d": states_demand[:, 3],
            "core_in_P_d": states_demand[:, 4],
            "core_out_P_d": states_demand[:, 5],
            "in_temp": in_temp,
            "out_temp": out_temp,
            "lb": lb_cons,
            "ub": ub_cons,
            "uc_in_temp": uc_in_temp,
            "uc_out_temp": uc_out_temp,
            "demand": requested_rho,
            "RL": actions.reshape(
                -1,
            ),
        }
        df = pd.DataFrame.from_dict(df)
        df.to_csv(dir_path + file_name + ".csv")


if __name__ == "__main__":
    MODEL_PATH = "./Data/Bebop/Data/Data/MLP_PPO_KNL_costDistance4_largeLR_initLag3_78-05-28-2023-17-09-47/"
    with open(MODEL_PATH + "config", "rb") as file:
        config = pickle.load(file)
    args = config["args"]
    agent_config = config["agent"]
    neurons = args.neurons
    layers = args.layers

    policy_config = {
        "action_dim": 2,
        "obs_dim": 16,
        "pi_layers": [neurons] * layers,
        "activation": tf.nn.tanh,
        "output_activation": None,
        "control_type": "continuous",
        "h_size": neurons,
        "batch_size": args.batch_size,
        "safety_layer": {"safety_level": 0},
    }

    evaluator = AgentEval(
        sindy_path="./env_data/red_SINDYc_model_2022-4-15-downsample.joblib",
        policy_model_weigths=MODEL_PATH + "model_weights_epoch_600",
        policy_config=policy_config,
        seed=35000,
        lstm_actor="False",
        if_red="False",
    )
    evaluator.rollout_trajectory(dir_path="./Figs/", file_name="test")
