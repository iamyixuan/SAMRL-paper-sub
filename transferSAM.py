# import modules
import psutil
import os
import gc
import pickle
import argparse
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpi4py import MPI
from datetime import datetime
from tqdm import tqdm

from lambda_ppo.model.lstm_ppo import PPOAgent as LSTMPPOAgent
from lambda_ppo.model.ppo import PPOAgent
from lambda_ppo.environment.ghfr_gym_rd import SINDyGYM
from lambda_ppo.model.mpi_utils import proc_id, mpi_avg
from lambda_ppo.environment.model_sindy_gfhr import sindyDx


def main(model_path):
    # load model configs with keys ['arch', 'agent', 'args']
    MODEL_PATH = model_path
    with open(MODEL_PATH + "config", "rb") as file:
        config = pickle.load(file)
    args = config["args"]
    agent_config = config["agent"]

    if proc_id() == 0:
        now = datetime.now()  # current date and time
        date_time = now.strftime("-%m-%d-%Y-%H-%M-%S")
        folder = args.expid + "_" + str(np.random.randint(0, 100)) + date_time
        path = "./Data/" + folder
        os.mkdir(path)

    # create the enviornment - to be replaced by the full SAM model.
    env = SINDyGYM(
        hist_len=args.histlen, skip=5, rem_time=False, time_independent=True
    )  # skip 10 for the other model

    # define the model according to the configs
    control_type = "continuous"
    obs_dim = env.state_dim
    act_dim = env.action_dim * 2

    if args.safety_level > 0:
        args.learn_cost_critic = False

    if args.learn_cost_critic:
        rew_dim = 3
    else:
        rew_dim = 1
    delta = args.delta

    seed = proc_id()

    neurons = args.neurons
    layers = args.layers
    mlp_arch = {
        "pi_layers": [neurons] * layers,
        "v_repr_layers": [],
        "v_layers": [neurons] * layers,
        "h_size": neurons,
        "batch_size": args.batch_size,
    }
    safety_layer = {"safety_level": args.safety_level}
    if safety_layer["safety_level"] > 0:
        dynamics_model = sindyDx()
        safety_layer["dynamics_model"] = dynamics_model

    if args.lstm_actor:
        h_size = mlp_arch["h_size"]
        batch_size = mlp_arch["batch_size"]
        trace_len = args.env_steps // batch_size
        agent = LSTMPPOAgent(
            obs_dim,
            act_dim,
            rew_dim,
            safety_layer=safety_layer,
            control_type=control_type,
            seed=seed,
            mlp_arch=mlp_arch,
            agent_config=agent_config,
            const_std=args.const_std,
            std_lb=args.std_lb,
        )

    else:
        agent = PPOAgent(
            obs_dim,
            act_dim,
            rew_dim,
            safety_layer=safety_layer,
            control_type=control_type,
            seed=seed,
            mlp_arch=mlp_arch,
            agent_config=agent_config,
            const_std=args.const_std,
            std_lb=args.std_lb,
        )

    # build models
    if args.lstm_actor:
        agent.policy_model.build(input_shape=[None, trace_len, obs_dim])
        agent.value_model.build(input_shape=[None, trace_len, obs_dim])
    else:
        agent.policy_model.build(input_shape=[None, obs_dim])
        agent.value_model.build(input_shape=[None, obs_dim])

    policy_weights = agent.policy_model.trainable_weights
    value_weights = agent.value_model.trainable_weights
    lagrange_multipliers = tf.Variable(
        initial_value=np.asarray([[args.lam_init, args.lam_init]]),
        trainable=True,
        dtype=tf.float32,
        constraint=lambda x: tf.clip_by_value(x, 0.0, np.inf),
    )

    zero_grads_policy = [tf.zeros_like(w) for w in policy_weights]
    zero_grads_value = [tf.zeros_like(w) for w in value_weights]
    # zero_grads_lag = [tf.zeros_like(w) for w in lagrange_multipliers]
    zero_grads_lag = [tf.zeros_like(lagrange_multipliers)]

    agent.optimizer_pi.apply_gradients(zip(zero_grads_policy, policy_weights))
    agent.optimizer_v.apply_gradients(zip(zero_grads_value, value_weights))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lam_lr, decay_steps=10000, decay_rate=0.9
    )
    optimizer_lagrange = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer_lagrange.apply_gradients(zip(zero_grads_lag, [lagrange_multipliers]))

    # load model weights and learned Lagrangians
    SAVED_W = "model_weights_epoch_600"
    with open(MODEL_PATH + SAVED_W, "rb") as file:
        weights = pickle.load(file)

    # load optimizer states. This step must be prior to loading the model weights. See more https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

    agent.optimizer_pi.set_weights(weights["pi_opt_state"])
    agent.optimizer_v.set_weights(weights["v_opt_state"])
    optimizer_lagrange.set_weights(weights["lag_opt_state"])

    agent.policy_model.set_weights(weights["policy"])
    agent.value_model.set_weights(weights["value"])

    lags = weights["lags"].ravel()
    lagrange_multipliers.assign(
        np.asarray([lags[0], lags[1]]).reshape(1, -1)
    )  # Replace None with the saved Lagrangians.

    # training

    # continued training begins...
    epochs = agent.agent_config["epochs"]
    steps_per_epoch = agent.agent_config["steps_per_epoch"]
    max_ep_len = agent.agent_config["max_ep_len"]

    save_freq = args.savefreq

    if proc_id() == 0:
        config = dict()
        config["arch"] = mlp_arch
        config["agent"] = agent_config
        config["args"] = args
        with open(path + "/config", "wb") as file:
            pickle.dump(config, file)

    o, d, ep_ret, ep_len = (
        env.reset(seed=proc_id()),
        False,
        np.asarray([0.0, 0.0, 0.0]),
        0,
    )
    print("init state shape", o.shape)

    # sample another state for validation
    # o_val = val_env.reset()
    # val_ret = np.array([0., 0., 0.])

    if args.lstm_actor:
        # Reset the recurrent layer's hidden state, 64 is number of output nodes in lstm
        piLSTM_state = (tf.zeros((1, h_size)), tf.zeros((1, h_size)))
        vLSTM_state = (tf.zeros((1, h_size)), tf.zeros((1, h_size)))
        # val_piLSTM_state =  (tf.zeros((1, h_size)), tf.zeros((1, h_size)))

    logger = dict()
    logger["pi_loss"] = []
    logger["kl_div"] = []
    logger["mse_loss"] = []
    logger["ep_ret"] = []
    logger["ep_len"] = []
    logger["lagrange"] = []
    logger["entropy"] = []

    # syncing initial policy weights across actors. note that while env is deterministic, policy is stochastic
    o = np.asarray(o, dtype=np.float32).reshape(1, -1)  # coming from the env reset()
    cons_bounds = np.asarray(env.cons_bounds()).reshape(1, -1)
    if args.lstm_actor:
        agent.output_actions(
            state=o.reshape(1, 1, -1), cons_bounds=cons_bounds, lstm_state=piLSTM_state
        )
    else:
        agent.output_actions(
            state=o, cons_bounds=cons_bounds
        )  # return the action and lop-probability of the action.

    if proc_id() == 0:
        weights_pi = agent.policy_model.get_weights()
        weights_v = agent.value_model.get_weights()
    else:
        weights_pi = None
        weights_v = None
    weights_pi = MPI.COMM_WORLD.bcast(weights_pi, root=0)
    weights_v = MPI.COMM_WORLD.bcast(weights_v, root=0)
    agent.policy_model.set_weights(weights_pi)
    agent.value_model.set_weights(weights_v)

    EXPID = args.expid
    logger["lagrange"].append(lagrange_multipliers.numpy().ravel())
    ###############################
    # Main loop: collect experience in env and update/log each epoch
    stime = time.time()
    # best_reward = -np.inf
    for epoch in range(epochs):
        # print("Epoch {}".format(epoch))
        epoch_ep_ret = []
        epoch_ep_len = []
        epoch_entropy = []
        # epoch_val_ret = []
        if not args.learn_cost_critic:
            state_costs = []
            epoch_costs = []
        if safety_layer["safety_level"] > 0:
            cons_bounds_arr = []
        pi_lstm_states = []
        v_lstm_states = []
        for t in tqdm(range(steps_per_epoch)):
            if args.lstm_actor:
                if t % trace_len == 0:
                    pi_lstm_states.append(piLSTM_state)
                    v_lstm_states.append(vLSTM_state)
            o = np.asarray(o, dtype=np.float32).reshape(1, -1)
            # o_val = np.asarray(o_val, dtype=np.float32).reshape(1,-1)
            if args.lstm_actor:
                if safety_layer["safety_level"] == 0:
                    a, logp_t, piLSTM_state, entropy = agent.output_actions(
                        state=o.reshape(1, 1, -1), lstm_state=piLSTM_state
                    )
                    # a_val, val_logp_t, val_piLSTM_state = agent.output_actions(state=o_val.reshape(1,1,-1),
                    # lstm_state=val_piLSTM_state)

                else:
                    cons_bounds = env.cons_bounds()
                    cons_bounds_arr.append(cons_bounds)
                    a, logp_t, piLSTM_state, entropy = agent.output_actions(
                        state=o.reshape(1, 1, -1),
                        cons_bounds=cons_bounds.reshape(1, -1),
                        lstm_state=piLSTM_state,
                    )
                    # a_val, val_logp_t, val_piLSTM_state = agent.output_actions(state=o_val.reshape(1,1,-1), cons_bounds=cons_bounds.reshape(1,-1),
                    #                                     lstm_state=val_piLSTM_state)

                v_t, vLSTM_state = agent.value_model(
                    o.reshape(1, 1, -1), lstm_state=vLSTM_state
                )
                v_t = v_t.numpy().ravel()
            else:
                # if MLP use the action limiter
                a_hist = [tf.constant([[0.0]])]
                if safety_layer["safety_level"] == 0:
                    a_limit = tf.constant(
                        [[1.5 * 0.0016]]
                    )  # 1.5 times the initial change
                    a, logp_t, entropy = agent.output_actions(
                        state=o.reshape(1, -1)
                    )  # here can implement the action limiter: bounded by 1.5 times the initial decrease/increase
                    if args.limiter == "True":
                        if a - a_hist[-1] > a_limit:
                            a = a_hist[-1] + a_limit
                        elif a - a_hist[-1] < -a_limit:
                            a = a_hist[-1] - a_limit
                    a_hist.append(a)
                else:
                    cons_bounds = env.cons_bounds()
                    cons_bounds_arr.append(cons_bounds)
                    a, logp_t, entropy = agent.output_actions(
                        state=o.reshape(1, -1), cons_bounds=cons_bounds.reshape(1, -1)
                    )
                v_t = agent.value_model(o.reshape(1, -1)).numpy().ravel()

            o2, rew_vec, d = env.step(
                a[0].numpy()
            )  # get reward and new state from the env based on the output action.
            ep_ret += rew_vec
            ep_len += 1

            # validation rewards
            # o2_val, val_rew_vec, val_d = val_env.step(a_val[0].numpy())
            # val_ret += val_rew_vec
            # primal dual approach on penalized reward function
            if safety_layer["safety_level"] == 0:
                if args.learn_cost_critic:
                    rew_vec[0] = (
                        rew_vec[0]
                        - lagrange_multipliers.numpy()[0][0] * rew_vec[1]
                        - lagrange_multipliers.numpy()[0][1] * rew_vec[2]
                    )
                    penalized_rew = rew_vec
                else:
                    state_costs.append(rew_vec[1:])
                    penalized_rew = np.asarray(
                        [
                            rew_vec[0]
                            - lagrange_multipliers.numpy()[0][0] * rew_vec[1]
                            - lagrange_multipliers.numpy()[0][1] * rew_vec[2]
                        ]
                    )
            else:
                penalized_rew = rew_vec[0]

            # save and log
            agent.buf.store(o, a, penalized_rew, v_t, logp_t.numpy())

            # Update obs (critical!)
            o = o2
            # o_val = o2_val

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not (terminal):
                    print("Warning: trajectory cut off by epoch at %d steps." % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target

                if args.learn_cost_critic:
                    if args.lstm_actor:
                        last_val = (
                            rew_vec.reshape(1, -1)
                            if d
                            else tf.squeeze(
                                agent.value_model(o.reshape(1, 1, -1), vLSTM_state)[0],
                                axis=-1,
                            ).numpy()
                        )
                    else:
                        last_val = (
                            rew_vec.reshape(1, -1)
                            if d
                            else tf.squeeze(
                                agent.value_model(o.reshape(1, -1)), axis=-1
                            ).numpy()
                        )
                else:
                    if args.lstm_actor:
                        last_val = (
                            rew_vec
                            if d
                            else tf.squeeze(
                                agent.value_model(o.reshape(1, 1, -1), vLSTM_state)[0],
                                axis=-1,
                            ).numpy()
                        )
                    else:
                        last_val = (
                            rew_vec
                            if d
                            else tf.squeeze(
                                agent.value_model(o.reshape(1, -1)), axis=-1
                            ).numpy()
                        )

                    if d:
                        if safety_layer["safety_level"] == 0:
                            last_val = np.asarray(
                                [
                                    last_val[0]
                                    - lagrange_multipliers.numpy()[0][0] * last_val[1]
                                    - lagrange_multipliers.numpy()[0][1] * last_val[2]
                                ]
                            ).reshape(1, -1)
                        else:
                            last_val = rew_vec[0]
                            last_val = np.asarray([[last_val]])
                agent.buf.finish_path(last_val)
                if args.lstm_actor:
                    # Reset the recurrent layer's hidden state
                    piLSTM_state = (tf.zeros((1, h_size)), tf.zeros((1, h_size)))
                    vLSTM_state = (tf.zeros((1, h_size)), tf.zeros((1, h_size)))
                    # val_piLSTM_state =  (tf.zeros((1, h_size)), tf.zeros((1, h_size)))
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    epoch_ep_ret.append(ep_ret)
                    epoch_ep_len.append(ep_len)
                    epoch_entropy.append(entropy)
                    # epoch_val_ret.append(val_ret)
                    if not args.learn_cost_critic and safety_layer["safety_level"] == 0:
                        state_costs = np.asarray(state_costs)
                        cost1_grad = np.mean(
                            agent.buf.discount_cumsum(
                                state_costs[:, 0], discount=args.gamma
                            )
                        )
                        cost2_grad = np.mean(
                            agent.buf.discount_cumsum(
                                state_costs[:, 1], discount=args.gamma
                            )
                        )
                        epoch_costs.append(np.asarray([cost1_grad, cost2_grad]))

                if not args.learn_cost_critic and safety_layer["safety_level"] == 0:
                    state_costs = []
                ep_ret, ep_len = np.asarray([0.0, 0.0, 0.0]), 0
                o, d = env.reset(seed=epoch + proc_id()), False
                # o_val = val_env.reset()

        # Perform PPO update!
        data = agent.buf.get()
        inputs = dict()
        if args.lstm_actor:
            inputs["state"] = data[0].reshape(batch_size, trace_len, obs_dim)
        else:
            inputs["state"] = data[0]
        inputs["action"] = data[1]
        inputs["advs"] = data[2]
        inputs["logp"] = data[-1]
        inputs["ret"] = data[3]
        if safety_layer["safety_level"] > 0:
            inputs["cons_bounds"] = np.asarray(cons_bounds_arr).reshape(1, -1)

        if args.lstm_actor:
            pi_hidden_states = tf.concat(
                [pi_lstm_state[0] for pi_lstm_state in pi_lstm_states], axis=0
            )
            pi_cell_states = tf.concat(
                [pi_lstm_state[1] for pi_lstm_state in pi_lstm_states], axis=0
            )
            v_hidden_states = tf.concat(
                [v_lstm_state[0] for v_lstm_state in v_lstm_states], axis=0
            )
            v_cell_states = tf.concat(
                [v_lstm_state[1] for v_lstm_state in v_lstm_states], axis=0
            )
            pi_lstm_states = (pi_hidden_states, pi_cell_states)
            v_lstm_states = (v_hidden_states, v_cell_states)

        mean_return = np.mean(epoch_ep_ret, axis=0)
        mean_return = mpi_avg(mean_return)
        mean_return = np.asarray(mean_return)

        mean_entropy = np.mean(epoch_entropy, axis=0)
        mean_entropy = mpi_avg(mean_entropy)
        mean_entropy = np.asarray(mean_entropy)

        # mean_val_return = np.mean(epoch_val_ret, axis=0)
        # mean_val_return = mpi_avg(mean_val_return)
        # mean_val_return = np.asarray(mean_val_return)

        if safety_layer["safety_level"] == 0:
            if args.learn_cost_critic:
                # implementing primal dual variable approach
                # notice how we sliced dim 0 to be 0:1, because we want the initial state distribution, not the steady-state distribution
                # slicing to get initial distribution yields worse policy. use steady state distr instead
                if args.lstm_actor:
                    critic_cost = tf.squeeze(
                        agent.value_model(inputs["state"], lstm_state=v_lstm_states)[0]
                    )[:, 1:]
                else:
                    critic_cost = tf.squeeze(agent.value_model(inputs["state"]))[
                        :, 1:
                    ]  # [:,1:]
                # clipping analyzed in https://arxiv.org/pdf/2205.11814.pdf
                cost_function = tf.clip_by_value(
                    tf.reshape(
                        tf.reduce_mean(delta - critic_cost, axis=0), shape=(-1, 1)
                    ),
                    -np.inf,
                    0.0,
                )
                cost_function = mpi_avg(cost_function)
            else:
                epoch_costs = np.mean(epoch_costs, axis=0)
                # implementing primal dual variable approach
                # clip by value is from your JSAC paper
                cost_function = tf.clip_by_value(
                    tf.reshape(
                        tf.convert_to_tensor(delta - epoch_costs), shape=(-1, 1)
                    ),
                    -np.inf,
                    0.0,
                )
                cost_function = mpi_avg(cost_function)
            with tf.GradientTape() as tape:
                lag_loss = tf.matmul(lagrange_multipliers, cost_function)
            # this gradient is actually your dO(u)/du gradient in your TNNLS paper.
            lag_gradients = tape.gradient(lag_loss, lagrange_multipliers)
            optimizer_lagrange.apply_gradients(
                zip([lag_gradients], [lagrange_multipliers])
            )
            ##############################################################

        mean_length = np.mean(epoch_ep_len)
        mean_length = mpi_avg(mean_length)
        if args.lstm_actor:
            loss_pi, approx_kl, mse = agent.update(
                inputs, pi_lstm_states, v_lstm_states
            )
        else:
            loss_pi, approx_kl, mse = agent.update(inputs)
        logger["pi_loss"].append(loss_pi)
        logger["kl_div"].append(approx_kl)
        logger["mse_loss"].append(mse)
        logger["ep_ret"].append(mean_return)
        logger["ep_len"].append(mean_length)
        logger["entropy"].append(mean_entropy)
        logger["lagrange"].append(lagrange_multipliers.numpy().ravel())
        # logger['val_ret'].append(mean_val_return)

        if proc_id() == 0:
            if safety_layer["safety_level"] == 0:
                print(
                    "Time {:10.2f}, Epoch {}, Ep Return {}, Lagrange {}, Grad {}".format(
                        time.time() - stime,
                        epoch,
                        mean_return,
                        lagrange_multipliers.numpy(),
                        cost_function.ravel(),
                    ),
                    flush=True,
                )
            else:
                print(
                    "Time {:10.2f}, Epoch {}, Ep Return {}".format(
                        time.time() - stime, epoch, mean_return
                    ),
                    flush=True,
                )

        if (epoch + 1) % save_freq == 0 and proc_id() == 0:
            print("Saving weights and states...")
            weights_policy = agent.policy_model.get_weights()
            weigths_val = agent.value_model.get_weights()  # save the value network
            pi_opt_state = agent.optimizer_pi.get_weights()
            v_opt_state = agent.optimizer_v.get_weights()
            lag_opt_state = optimizer_lagrange.get_weights()
            weights = {
                "policy": weights_policy,
                "value": weigths_val,
                "pi_opt_state": pi_opt_state,
                "v_opt_state": v_opt_state,
                "lag_opt_state": lag_opt_state,
                "lags": lagrange_multipliers.numpy(),
            }
            with open(path + "/model_weights_epoch_" + str(epoch + 1), "wb") as file:
                pickle.dump(weights, file)
            with open(path + "/logger", "wb") as file:
                pickle.dump(logger, file)

    if proc_id() == 0:
        log_data = np.asarray(logger["ep_ret"])
        fig, axis = plt.subplots(2, 2, figsize=(10, 10))
        axis[0, 0].plot(log_data[:, 0])
        axis[0, 0].set_title("Rewards")
        axis[0, 1].plot(log_data[:, 1])
        axis[0, 1].set_title("Cost 1")
        axis[1, 0].plot(log_data[:, 2])
        axis[1, 0].set_title("Cost 2")
        lagrange_data = np.asarray(logger["lagrange"])
        axis[1, 1].plot(lagrange_data[:, 0], label="Lagrange mul 1")
        axis[1, 1].plot(lagrange_data[:, 1], label="Lagrange mul 2")
        axis[1, 1].legend()
        plt.savefig(path + "/learning_curves.pdf")
    # save final neural network weights
    if proc_id() == 0:
        weights_policy = agent.policy_model.get_weights()
        weigths_val = agent.value_model.get_weights()  # save the value network
        weights = {"policy": weights_policy, "value": weigths_val}
        with open(path + "/model_weights_final", "wb") as file:
            pickle.dump(weights, file)
        with open(path + "/logger", "wb") as file:
            pickle.dump(logger, file)


if __name__ == "__main__":
    # replace the model folder path.
    main(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_costDistance4_largeLR_initLag3_78-05-28-2023-17-09-47/"
    )
