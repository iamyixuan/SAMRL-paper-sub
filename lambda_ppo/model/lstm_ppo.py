import tensorflow as tf
import numpy as np
import os
from .lstm_actor import Actor
from .lstm_critic import Critic
import scipy.signal
from .mpi_utils import mpi_avg, mpi_sum

default_config = {
    "steps_per_epoch": 2000,  # 1600, cartpole
    "max_ep_len": 2000,  # 200,
    "epochs": 400,  # 20,
    "gamma": 0.99,
    "clip_ratio": 0.1,
    "pi_lr": 1e-4,
    "vf_lr": 1e-3,
    "train_pi_iters": 80,
    "train_v_iters": 80,
    "lam": 0.95,
    "target_kl": 0.01,
}


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        size,
        rew_dim,
        gamma=0.99,
        lam=0.95,
        control_type="discrete",
    ):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if control_type == "discrete":
            self.act_buf = np.zeros((size,), dtype=np.int32)
        else:
            self.act_buf = np.zeros(
                (size, act_dim // 2), dtype=np.float32
            )  # because we are parameterizing mean and std
        self.adv_buf = np.zeros((size, rew_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, rew_dim), dtype=np.float32)
        self.ret_buf = np.zeros((size, rew_dim), dtype=np.float32)
        self.val_buf = np.zeros((size, rew_dim), dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.size = size
        self.rew_dim = rew_dim

        self.sub_ep_start_idx = 0
        self.sub_ep_rew = []
        self.sub_ep_val = []

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
     
    def done(self, last_val):
        """
        Only call this method when breaking down the long episodes
        """
        path_slice = slice(self.sub_ep_start_idx, self.ptr)
        rew_buf_tmp = np.append(self.rew_buf[path_slice], last_val, axis=0)
        val_buf_tmp = np.append(self.val_buf[path_slice], last_val, axis=0) 
        self.sub_ep_rew.append(rew_buf_tmp)
        self.sub_ep_val.append(val_buf_tmp)
        self.sub_ep_start_idx += 2250

    def finish_path(self, last_val):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        if os.environ.get("BOOTSTRAP_FOR_LONG_EPISODE") == "True":
            print("Bootstrapping...")
            # Reshape to calculate reward-to-go and advantage for split epsidoes
            # rews = self.rew_buf
            # vals = self.val_buf
            # rews = rews.reshape(self.size//2250, 2251, self.rew_dim) # last value is bootstrapped
            # vals = vals.reshape(self.size//2250, 2251, self.rew_dim)

            rews = np.asarray(self.sub_ep_rew)
            vals = np.asarray(self.sub_ep_val)

            self.adv_buf = self.adv_buf.reshape(self.size//2250, 2250, self.rew_dim)
            self.ret_buf = self.rew_buf.reshape(self.size//2250, 2250, self.rew_dim)

            path_slice = slice(self.path_start_idx, int(self.ptr/5))
            deltas = rews[:,:-1, :] + self.gamma * vals[:, 1:, :] - vals[:, :-1, :]
            self.adv_buf[:,path_slice,:] = self.discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[:, path_slice,:] = self.discount_cumsum(rews, self.gamma)[:,:-1,:]

            # Reshape back to original for feeding data into networks
            self.ret_buf = self.rew_buf.reshape(self.size, self.rew_dim)
            self.adv_buf = self.adv_buf.reshape(self.size, self.rew_dim)

            self.path_start_idx = self.ptr
            self.sub_ep_start_idx = 0
            self.sub_ep_rew = []
            self.sub_ep_val = []

            
        else:
            path_slice = slice(self.path_start_idx, self.ptr)
            rews = np.append(self.rew_buf[path_slice], last_val, axis=0)
            vals = np.append(self.val_buf[path_slice], last_val, axis=0)

            # 1the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
            self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick

        np_adv_mean = []
        np_adv_std = []
        for arr in self.adv_buf.T:
            global_sum, global_n = mpi_sum([np.sum(arr), len(self.adv_buf)])
            adv_mean = global_sum / global_n
            np_adv_mean.append(adv_mean)

            global_sum_sq = mpi_sum(np.sum((arr - adv_mean) ** 2))
            adv_std = np.sqrt(global_sum_sq / global_n)  # compute global std
            np_adv_std.append(adv_std)

        adv_mean = np.asarray(np_adv_mean)
        adv_std = np.asarray(np_adv_std) + 1e-8
        # adv_mean = np.mean(self.adv_buf)
        # adv_std = np.std(self.adv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        if os.environ.get("BOOTSTRAP_FOR_LONG_EPISODE") == "True":
            return scipy.signal.lfilter([1], [1, float(-discount)], x[:, ::-1, :], axis=1)[:, ::-1, :]
        else:
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        rew_dim,
        safety_layer,
        const_std,
        std_lb,
        control_type="discrete",
        agent_config=None,
        seed=0,
        mlp_arch={
            "pi_layers": [64, 64, 64],
            "v_repr_layers": [64, 64],
            "v_layers": [64, 64],
            "h_size": 64,
            "batch_size": 50,
        },
    ):
        if agent_config is None:
            self.agent_config = default_config
        else:
            self.agent_config = agent_config
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.steps_per_epoch = self.agent_config["steps_per_epoch"]
        self.epochs = self.agent_config["epochs"]
        self.gamma = self.agent_config["gamma"]
        self.clip_ratio = self.agent_config["clip_ratio"]
        self.pi_lr = self.agent_config["pi_lr"]
        self.vf_lr = self.agent_config["vf_lr"]
        self.train_pi_iters = self.agent_config["train_pi_iters"]
        self.train_v_iters = self.agent_config["train_v_iters"]
        self.lam = self.agent_config["lam"]
        self.max_ep_len = self.agent_config["max_ep_len"]
        self.target_kl = self.agent_config["target_kl"]
        self.entrop_coeff = self.agent_config["entrop_coeff"]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rew_dim = rew_dim
        self.buf = PPOBuffer(
            obs_dim,
            act_dim,
            self.steps_per_epoch,
            rew_dim,
            self.gamma,
            self.lam,
            control_type,
        )
        pi_layers, v_repr_layers, v_layers = (
            mlp_arch["pi_layers"],
            mlp_arch["v_repr_layers"],
            mlp_arch["v_layers"],
        )
        self.h_size, self.batch_size = mlp_arch["h_size"], mlp_arch["batch_size"]
        if os.environ.get("BOOTSTRAP_FOR_LONG_EPISODE") == "True":
            print("Bootstrapping...")
            self.batch_size = self.batch_size//5
        self.build_agent(
            control_type,
            pi_layers,
            v_repr_layers,
            v_layers,
            self.h_size,
            self.batch_size,
            safety_layer=safety_layer,
            const_std=const_std,
            std_lb=std_lb,
        )

    @tf.function
    def pi_loss(self, logp, logp_old_ph, adv_ph, clip_ratio, entropy):
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        # this is an equivalent objective to the one found in the paper
        min_adv = tf.where(
            adv_ph[:, 0] > 0,
            (1 + clip_ratio) * adv_ph[:, 0],
            (1 - clip_ratio) * adv_ph[:, 0],
        )
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph[:, 0], min_adv), axis=0)
        return pi_loss - self.entrop_coeff * tf.reduce_mean(entropy)

        """
        ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
        ratio = tf.stack([ratio]*adv_ph.shape[1],axis=1)
        #this is an equivalent objective to the one found in the paper
        min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv),axis=0)
        return pi_loss[0] - self.entrop_coeff * tf.reduce_mean(entropy)

        pi_loss = tf.reshape(pi_loss,shape=(-1,1))
        weights = tf.reshape(tf.convert_to_tensor(weights),shape=(1,-1))
        return tf.squeeze(tf.matmul(weights,pi_loss),axis=1)

        #random alternation
        #return pi_loss[np.random.randint(0,3)]
        #approx_ent = tf.reduce_mean(-logp)
        """

    @tf.function
    def value_loss(self, ret_ph, v):
        # returning a 3x1 vector or the mean of it yield the same results on gradient computation
        return tf.reduce_mean(tf.square(ret_ph - v))

    @tf.function  # (because we are doing mpi op over numpy arrays, we send back gradients)
    def train_pi_step(self, inputs, lstm_state):
        states = inputs["state"]
        actions = inputs["action"]
        advs = inputs["advs"]
        logp_old = inputs["logp"]
        if "cons_bounds" in inputs:
            cons_bounds = tf.cast(inputs["cons_bounds"], dtype=tf.float32)
        else:
            cons_bounds = None
        with tf.GradientTape() as tape:
            target_logp, entropy, cell_state = self.policy_model.policy_train(
                obs_ph=states,
                action_ph=actions,
                cons_bounds=cons_bounds,
                training=True,
                lstm_state=lstm_state,
            )
            loss = self.pi_loss(target_logp, logp_old, advs, self.clip_ratio, entropy)
        approx_kl = tf.reduce_mean(
            logp_old - target_logp
        )  # a sample estimate for KL-divergence, easy to compute
        gradients = tape.gradient(loss, self.policy_model.trainable_variables)
        return loss, approx_kl, gradients, cell_state

    @tf.function
    def train_v_step(self, inputs, lstm_state):
        ret = inputs["ret"]
        states = inputs["state"]
        with tf.GradientTape() as tape:
            v, cell_state = self.value_model(
                states, lstm_state=lstm_state, training=True
            )
            v = tf.squeeze(v)
            mse = self.value_loss(v=v, ret_ph=ret)
        gradients = tape.gradient(mse, self.value_model.trainable_variables)
        return mse, gradients, cell_state

    def update(self, inputs, pi_lstm, v_lstm):
        # Training
        loss_pi_list = []
        approx_kl_list = []
        loss_v_list_mse = []
        lstm_state = pi_lstm
        for i in range(self.train_pi_iters):
            loss_pi, approx_kl, pi_gradients, lstm_state = self.train_pi_step(
                inputs, lstm_state=lstm_state
            )
            hidden_state, cell_state = lstm_state[0], lstm_state[1]
            hidden_state = tf.concat((tf.zeros((1, self.h_size)), hidden_state), axis=0)
            cell_state = tf.concat((tf.zeros((1, self.h_size)), cell_state), axis=0)
            hidden_state = tf.slice(
                hidden_state, [0, 0], [self.batch_size, self.h_size]
            )
            cell_state = tf.slice(cell_state, [0, 0], [self.batch_size, self.h_size])
            lstm_state = (hidden_state, cell_state)
            np_gradients = []
            for grad in pi_gradients:
                np_grad = grad.numpy()
                np_grad = mpi_avg(np_grad)
                np_gradients.append(np_grad)
            self.optimizer_pi.apply_gradients(
                zip(np_gradients, self.policy_model.trainable_variables)
            )
            approx_kl = mpi_avg(approx_kl)
            loss_pi = mpi_avg(loss_pi)
            loss_pi_list.append(loss_pi)
            approx_kl_list.append(approx_kl)
            if approx_kl > 1.5 * self.target_kl:
                print("Early stopping at step %d due to reaching max kl." % i)
                break
        lstm_state = v_lstm
        for _ in range(self.train_v_iters):
            mse, v_gradients, lstm_state = self.train_v_step(
                inputs, lstm_state=lstm_state
            )
            hidden_state, cell_state = lstm_state[0], lstm_state[1]
            hidden_state = tf.concat((tf.zeros((1, self.h_size)), hidden_state), axis=0)
            cell_state = tf.concat((tf.zeros((1, self.h_size)), cell_state), axis=0)
            hidden_state = tf.slice(
                hidden_state, [0, 0], [self.batch_size, self.h_size]
            )
            cell_state = tf.slice(cell_state, [0, 0], [self.batch_size, self.h_size])
            lstm_state = (hidden_state, cell_state)
            np_gradients = []
            for grad in v_gradients:
                np_grad = grad.numpy()
                np_grad = mpi_avg(np_grad)
                np_gradients.append(np_grad)
            self.optimizer_v.apply_gradients(
                zip(np_gradients, self.value_model.trainable_variables)
            )
            mse = mpi_avg(mse)
            loss_v_list_mse.append(mse)
        return np.mean(loss_pi_list), np.mean(approx_kl_list), np.mean(loss_v_list_mse)

    def output_actions(
        self, state, cons_bounds=None, deterministic=False, lstm_state=None
    ):
        action, logp_action, lstm_state, entropy = self.policy_model.policy_act(
            state,
            cons_bounds,
            training=False,
            deterministic=deterministic,
            lstm_state=lstm_state,
        )
        return action, tf.squeeze(logp_action), lstm_state, entropy

    def build_agent(
        self,
        control_type,
        pi_layers,
        v_repr_layers,
        v_layers,
        h_size,
        batch_size,
        safety_layer,
        const_std,
        std_lb,
    ):
        if const_std == "True":
            print("Use constant policy standard deviation...")
        self.policy_model = Actor(
            action_dim=self.act_dim,
            hidden_sizes=pi_layers,
            activation=tf.nn.tanh,
            output_activation=None,
            control_type=control_type,
            h_size=h_size,
            batch_size=batch_size,
            safety_layer=safety_layer,
            const_std=const_std,
            std_lb=std_lb,
        )
        self.value_model = Critic(
            output_dim=self.rew_dim,
            repr_sizes=v_repr_layers,
            hidden_sizes=v_layers,
            activation=tf.nn.tanh,
            output_activation=None,
            h_size=h_size,
            batch_size=batch_size,
        )
        self.optimizer_pi = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=self.vf_lr)
