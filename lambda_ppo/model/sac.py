import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from .actor import Actor
from .critic import Critic
from .mpi_utils import mpi_avg


default_config = {
    "steps_per_epoch": 2250,  # 1600, cartpole
    "max_ep_len": 2250,  # 200,
    "epochs": 400,  # 20,
    "gamma": 0.99,
    "pi_lr": 1e-4,
    "q_lr": 1e-3,
    "replay_size": int(1e6),
    "lam": 0.95,
    "alpha": 1,
    "polyak": 0.995,
    "start_steps": 2250 * 10,
    "std_attenuate": 1,
}


class ReplayBuffer:
    """
    A buffer for storing agent experience
    """

    def __init__(self, obs_dim, act_dim, size, rew_dim, control_type="discrete"):
        self.obs1_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if control_type == "discrete":
            self.acts_buf = np.zeros((size,), dtype=np.int32)
        else:
            self.acts_buf = np.zeros(
                (size, act_dim // 2), dtype=np.float32
            )  # because we are parameterizing mean and std

        self.rews_buf = np.zeros((size, rew_dim), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        # we are only supporting one constraint
        self.current_cons_bounds_buf = np.zeros((size, 1), dtype=np.float32)
        self.next_cons_bounds_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        obs,
        act,
        rew,
        next_obs,
        done,
        current_cons_bounds=None,
        next_cons_bounds=None,
    ):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if current_cons_bounds is not None:
            self.current_cons_bounds_buf[self.ptr] = current_cons_bounds
        if next_cons_bounds is not None:
            self.next_cons_bounds_buf[self.ptr] = next_cons_bounds
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            current_cons_bounds=self.current_cons_bounds_buf[idxs],
            next_cons_bounds=self.next_cons_bounds_buf[idxs],
        )


class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        rew_dim,
        safety_layer,
        control_type="discrete",
        agent_config=None,
        seed=0,
        mlp_arch={
            "pi_layers": [64, 64, 64],
            "v_repr_layers": [64, 64],
            "v_layers": [64, 64],
        },
    ):
        if agent_config is None:
            self.agent_config = default_config
        else:
            self.agent_config = agent_config
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rew_dim = rew_dim
        self.steps_per_epoch = self.agent_config["steps_per_epoch"]
        self.epochs = self.agent_config["epochs"]
        self.gamma = self.agent_config["gamma"]
        self.pi_lr = self.agent_config["pi_lr"]
        self.q_lr = self.agent_config["q_lr"]
        pi_layers, v_repr_layers, v_layers = (
            mlp_arch["pi_layers"],
            mlp_arch["v_repr_layers"],
            mlp_arch["v_layers"],
        )
        self.replay_size = self.agent_config["replay_size"]
        self.alpha = tf.constant(self.agent_config["alpha"])
        self.polyak = self.agent_config["polyak"]
        self.start_steps = self.agent_config["start_steps"]
        self.buf = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=self.replay_size,
            rew_dim=rew_dim,
            control_type=control_type,
        )
        self.build_agent(
            control_type, pi_layers, v_repr_layers, v_layers, safety_layer=safety_layer
        )

    @tf.function
    def train_step(self, data, lagrange):
        # bug: when we do tf.minimum, it has two be between two single dimensional vectors!!
        rewards = data["rews"]
        states = data["obs1"]
        next_states = data["obs2"]
        actions = data["acts"]
        dones = tf.reshape(data["done"], shape=(-1, 1))
        if "current_cons_bounds" in data:
            current_cons_bounds = tf.cast(data["current_cons_bounds"], dtype=tf.float32)
            next_cons_bounds = tf.cast(data["next_cons_bounds"], dtype=tf.float32)
        else:
            current_cons_bounds = None
            next_cons_bounds = None
        with tf.GradientTape(persistent=True) as tape:
            # next state action log probs
            # next state action and log probs
            next_mean, next_std, _ = self.policy_model.mlp_gaussian_policy_act(
                obs_ph=next_states,
                cons_bounds=next_cons_bounds,
                training=True,
                deterministic=True,
            )
            next_actions, logp_next_actions = self.process_actions(next_mean, next_std)

            # critics loss
            s_a = tf.concat((states, actions), axis=1)
            current_q_1 = tf.squeeze(self.q_1(s_a, training=True))
            current_q_2 = tf.squeeze(self.q_2(s_a, training=True))

            next_s_a = tf.concat((next_states, next_actions), axis=1)
            next_q_1 = tf.squeeze(self.q_1_target(next_s_a, training=True))
            next_q_2 = tf.squeeze(self.q_2_target(next_s_a, training=True))
            q_r = tf.math.minimum(next_q_1[:, 0:1], next_q_2[:, 0:1])
            q_c = tf.math.maximum(next_q_1[:, 1:3], next_q_2[:, 1:3])
            next_q_min = tf.concat((q_r, q_c), axis=1)

            state_values = next_q_min - self.alpha * logp_next_actions
            target_qs = tf.stop_gradient(
                rewards + state_values * self.gamma * (1.0 - dones)
            )
            critic_loss_1 = tf.reduce_mean(
                0.5 * tf.math.square(current_q_1 - target_qs)
            )
            critic_loss_2 = tf.reduce_mean(
                0.5 * tf.math.square(current_q_2 - target_qs)
            )
            # current state action log probs
            mean, std, _ = self.policy_model.mlp_gaussian_policy_act(
                obs_ph=states,
                cons_bounds=current_cons_bounds,
                training=True,
                deterministic=True,
            )
            actions, log_probs = self.process_actions(mean, std)
            s_a = tf.concat((states, actions), axis=1)

            current_q_1 = tf.squeeze(self.q_1(s_a, training=True))
            current_q_2 = tf.squeeze(self.q_2(s_a, training=True))
            current_q_1 = (
                current_q_1[:, 0:1]
                - lagrange[0][0] * current_q_1[:, 1:2]
                - lagrange[0][1] * current_q_1[:, 2:3]
            )
            current_q_2 = (
                current_q_2[:, 0:1]
                - lagrange[0][0] * current_q_2[:, 1:2]
                - lagrange[0][1] * current_q_2[:, 2:3]
            )

            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

        critic1_grad = tape.gradient(
            critic_loss_1, self.q_1.trainable_variables
        )  # compute critic 1 gradient
        critic2_grad = tape.gradient(
            critic_loss_2, self.q_2.trainable_variables
        )  # compute critic 2 gradient
        actor_grad = tape.gradient(
            actor_loss, self.policy_model.trainable_variables
        )  # compute actor gradient
        return (
            critic1_grad,
            critic2_grad,
            actor_grad,
            critic_loss_1 + critic_loss_2,
            actor_loss,
        )

    def update(self, data, lagrange):
        # Training
        (
            critic1_grad,
            critic2_grad,
            actor_grad,
            critic_loss,
            actor_loss,
        ) = self.train_step(data, lagrange)
        # all-reduce for critic 1 gradients
        np_gradients_q1 = []
        for grad in critic1_grad:
            np_grad = grad.numpy()
            np_grad = mpi_avg(np_grad)
            np_gradients_q1.append(np_grad)

        # all-reduce for critic 2 gradients
        np_gradients_q2 = []
        for grad in critic2_grad:
            np_grad = grad.numpy()
            np_grad = mpi_avg(np_grad)
            np_gradients_q2.append(np_grad)

        # all-reduce for actor gradients
        np_gradients_pi = []
        for grad in actor_grad:
            np_grad = grad.numpy()
            np_grad = mpi_avg(np_grad)
            np_gradients_pi.append(np_grad)

        self.update_models(np_gradients_q1, np_gradients_q2, np_gradients_pi)

        critic_loss = mpi_avg(critic_loss)
        actor_loss = mpi_avg(actor_loss)
        # very important... if you don't do this, its crap!!
        self.update_target_weights(model=self.q_1, target_model=self.q_1_target)
        self.update_target_weights(model=self.q_2, target_model=self.q_2_target)
        return critic_loss, actor_loss

    @tf.function
    def update_models(self, np_gradients_q1, np_gradients_q2, np_gradients_pi):
        self.optimizer_q1.apply_gradients(
            zip(np_gradients_q1, self.q_1.trainable_variables)
        )
        self.optimizer_q2.apply_gradients(
            zip(np_gradients_q2, self.q_2.trainable_variables)
        )
        self.optimizer_pi.apply_gradients(
            zip(np_gradients_pi, self.policy_model.trainable_variables)
        )

    @tf.function
    def process_actions(self, mean, std, eps=1e-6):

        raw_actions = mean

        raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float32) * std

        log_prob_u = tfp.distributions.MultivariateNormalDiag(
            loc=mean, scale_diag=std
        ).log_prob(raw_actions)
        log_prob_u = tf.reshape(log_prob_u, shape=(-1, 1))
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(
            log_prob_u
            - tf.math.log(
                self.clip_but_pass_gradient(1 - actions**2, l=0, u=1) + eps
            ),
            axis=1,
        )
        log_prob = tf.reshape(log_prob, shape=(-1, 1))
        return actions, log_prob

    @tf.function
    def clip_but_pass_gradient(self, x, l=-1.0, u=1.0):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)

    def output_actions(self, state, cons_bounds=None, deterministic=False):
        mean, std, logp_action = self.policy_model.policy_act(
            state, cons_bounds, training=False, deterministic=deterministic
        )
        return mean, std, logp_action

    def build_agent(
        self, control_type, pi_layers, v_repr_layers, v_layers, safety_layer
    ):
        self.policy_model = Actor(
            action_dim=self.act_dim,
            hidden_sizes=pi_layers,
            activation=tf.nn.tanh,
            output_activation=tf.nn.tanh,
            control_type=control_type,
            safety_layer=safety_layer,
        )
        self.q_1 = Critic(
            output_dim=self.rew_dim,
            repr_sizes=v_repr_layers,
            hidden_sizes=v_layers,
            activation=tf.nn.tanh,
            output_activation=None,
        )
        self.q_2 = Critic(
            output_dim=self.rew_dim,
            repr_sizes=v_repr_layers,
            hidden_sizes=v_layers,
            activation=tf.nn.tanh,
            output_activation=None,
        )
        self.q_1_target = Critic(
            output_dim=self.rew_dim,
            repr_sizes=v_repr_layers,
            hidden_sizes=v_layers,
            activation=tf.nn.tanh,
            output_activation=None,
        )
        self.q_2_target = Critic(
            output_dim=self.rew_dim,
            repr_sizes=v_repr_layers,
            hidden_sizes=v_layers,
            activation=tf.nn.tanh,
            output_activation=None,
        )
        self.optimizer_pi = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
        self.optimizer_q1 = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        self.optimizer_q2 = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        # sync main and target
        self.update_target_weights(model=self.q_1, target_model=self.q_1_target, tau=1)
        self.update_target_weights(model=self.q_2, target_model=self.q_2_target, tau=1)

    def update_target_weights(self, model, target_model, tau=None):
        if tau == None:
            tau = 1 - self.polyak
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(
            len(target_weights)
        ):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)
