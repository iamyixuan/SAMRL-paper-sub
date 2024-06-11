import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Actor(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        safety_layer={"safety_level": 0},
        hidden_sizes=(32,),
        activation=tf.nn.tanh,
        batch_size=50,
        h_size=64,
        output_activation=None,
        control_type="discrete",
        const_std="False",
        std_lb=0.01,
    ):
        super(Actor, self).__init__()
        self.model_layers = []
        self.activation = activation
        self.output_activation = output_activation
        self.hidden_sizes = hidden_sizes
        self.action_dim = action_dim
        self.control_type = control_type
        self.h_size = h_size
        self.batch_size = batch_size
        for h in self.hidden_sizes:
            self.model_layers.append(
                tf.keras.layers.Dense(
                    units=h,
                    activation=self.activation,
                    kernel_initializer=tf.keras.initializers.Orthogonal(
                        gain=tf.sqrt(2.0)
                    ),
                )
            )

        self.lstm1 = tf.keras.layers.LSTM(
            units=h_size,
            return_state=True,
            stateful=False,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
        self.model_layers.append(
            tf.keras.layers.Dense(
                units=self.action_dim,
                activation=self.output_activation,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
            )
        )
        if safety_layer["safety_level"] > 0:
            self.dynamics = safety_layer["dynamics_model"]
        if safety_layer["safety_level"] == 0:
            self.apply_safety_layer = self.identity_layer
        elif safety_layer["safety_level"] == 1:
            self.apply_safety_layer = self.safety_layer
        elif safety_layer["safety_level"] == 2:
            self.apply_safety_layer = self.stochastic_safety_layer
        self.state_dim = 5  # need to switch to 13 when doing SAM
        self.const_std = const_std
        self.std_lb = std_lb

    @tf.function
    def get_initial_zero_state(self, inputs):
        return (
            tf.zeros((self.batch_size, self.h_size)),
            tf.zeros((self.batch_size, self.h_size)),
        )

    # @tf.function #runtime error if we do so
    def get_initial_state(self, inputs):
        return (self.hidden_state, self.cell_state)

    @tf.function
    def call(self, x, lstm_state=None):
        # x = (state, lstm_state:(hidden, cell))
        # if lstm_state is None:
        #     self.lstm1.get_initial_state = self.get_initial_zero_state # might cause leak
        # else:
        #     self.hidden_state,self.cell_state = lstm_state[0],lstm_state[1]
        #     self.lstm1.get_initial_state = self.get_initial_state

        # x, lstm_state = x
        x = tf.cast(x, tf.float32)
        x, hidden_state, cell_state = self.lstm1(x, initial_state=lstm_state)
        for layer in self.model_layers[:-1]:
            x = layer(x)
        x = self.model_layers[-1](x)

        return x, (hidden_state, cell_state)

    @tf.function
    def policy_train(
        self,
        obs_ph,
        action_ph,
        cons_bounds=None,
        training=False,
        lstm_state=None,
        return_mean=False,
    ):
        if self.control_type == "discrete":
            return self.mlp_categorical_policy_train(
                obs_ph, action_ph, lstm_state=lstm_state, training=training
            )
        else:
            return self.mlp_gaussian_policy_train(
                obs_ph,
                action_ph,
                cons_bounds,
                lstm_state=lstm_state,
                training=training,
                return_mean=return_mean,
            )

    @tf.function
    def policy_act(
        self,
        obs_ph,
        cons_bounds=None,
        training=False,
        deterministic=False,
        lstm_state=None,
    ):
        if self.control_type == "discrete":
            return self.mlp_categorical_policy_act(
                obs_ph, training=training, lstm_state=lstm_state
            )
        else:
            return self.mlp_gaussian_policy_act(
                obs_ph,
                cons_bounds,
                training=training,
                deterministic=deterministic,
                lstm_state=lstm_state,
            )

    @tf.function
    def mlp_categorical_policy_act(self, obs_ph, training=False, lstm_state=None):
        logits, lstm_state = self(obs_ph, lstm_state=lstm_state, training=training)
        N = logits.shape[0]
        T = logits.shape[1]
        logits = tf.reshape(logits, shape=(N * T, logits.shape[2]))
        pi = tfp.distributions.Categorical(logits=logits)
        action = pi.sample()
        action_one_hot = tf.one_hot(action, depth=self.action_dim)
        """
        tf.nn.softmax(logits): logits -> tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        tf.nn.log_softmax: logits->log( tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis) )
        
        """
        logp_policy = tf.nn.log_softmax(logits)  # log probs over actions
        logp_action = tf.reduce_sum(
            tf.multiply(action_one_hot, logp_policy), axis=1
        )  # log prob of selected action

        return action, logp_action, lstm_state

    @tf.function
    def mlp_categorical_policy_train(
        self, obs_ph, action_ph, training=False, lstm_state=None
    ):
        logits, cell_state = self(obs_ph, lstm_state=lstm_state, training=training)
        N = logits.shape[0]
        T = logits.shape[1]
        logits = tf.reshape(logits, shape=(N * T, logits.shape[2]))
        logp_policy = tf.nn.log_softmax(logits)  # log probs over actions
        logp_action = tf.reduce_sum(
            tf.cast(tf.one_hot(action_ph, depth=self.action_dim), dtype=tf.float32)
            * logp_policy,
            axis=1,
        )
        # note: logp_action can also be found using (with necessary tensor shaping)
        # logp_action = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_ph)
        entropy = self.discrete_entropy(logits)
        return logp_action, entropy, cell_state

    @tf.function
    def discrete_entropy(self, logits):
        """
        entroy = -sum(p*log(p))
        """
        log_probs = tf.nn.log_softmax(logits)
        probs = tf.nn.softmax(logits)
        return -tf.reduce_sum(tf.multiply(probs, log_probs), axis=1)

    @tf.function
    def mlp_gaussian_policy_act(
        self,
        obs_ph,
        cons_bounds=None,
        training=False,
        deterministic=False,
        lstm_state=None,
    ):
        dist_params, lstm_state = self(obs_ph, lstm_state=lstm_state, training=training)
        N = dist_params.shape[0]
        T = dist_params.shape[1]
        dist_params = tf.reshape(dist_params, shape=(N * T, dist_params.shape[2]))
        action_size = dist_params.shape[1] // 2
        batch_size = dist_params.shape[0]
        assert 2 * action_size == dist_params.shape[1]
        # Note: in the new data set, we allow negative control signals, hence we need to make std positive.
        mean_fn = tf.slice(dist_params, [0, 0], [batch_size, action_size])
        # std_fn = tf.exp(tf.slice(dist_params, [0,action_size], [batch_size,action_size]))
        ## make sure the variance function can go to zero
        std_fn = (
            tf.slice(dist_params, [0, action_size], [batch_size, action_size]) + 1
        ) * 0.5
        std_lb = tf.ones(std_fn.shape) * self.std_lb  # set lower bound
        std_fn = tf.maximum(std_fn, std_lb)

        if self.const_std == "True":
            std_fn = tf.ones(std_fn.shape) * self.std_lb  # fix the standard deviation
        # factor = tf.math.pow(self.rate,tf.divide(action_count,self.decay_steps))
        # std_fn = tf.multiply(std_fn,tf.cast(factor,dtype=tf.float32))
        mean_fn, std_fn = self.apply_safety_layer(obs_ph, mean_fn, std_fn, cons_bounds)
        pi = tfp.distributions.MultivariateNormalDiag(loc=mean_fn, scale_diag=std_fn)
        action = pi.sample()
        logp_action = self.gaussian_log_p(action, mean_fn, std_fn)
        entropy = self.gaussian_entropy(std_fn)
        if not deterministic:
            return action, logp_action, lstm_state, entropy
        else:
            return mean_fn, std_fn, logp_action, lstm_state

    @tf.function
    def mlp_gaussian_policy_train(
        self,
        obs_ph,
        action_ph,
        cons_bounds=None,
        training=False,
        lstm_state=None,
        return_mean=False,
    ):
        dist_params, cell_state = self(obs_ph, lstm_state=lstm_state, training=training)
        N = dist_params.shape[0]
        T = dist_params.shape[1]
        dist_params = tf.reshape(dist_params, shape=(N * T, dist_params.shape[2]))
        action_size = dist_params.shape[1] // 2
        batch_size = dist_params.shape[0]
        assert 2 * action_size == dist_params.shape[1]
        # Note: in the new data set, we allow negative control signals, hence we need to make std positive.
        mean_fn = tf.slice(dist_params, [0, 0], [batch_size, action_size])
        # std_fn = tf.exp(tf.slice(dist_params, [0,action_size], [batch_size,action_size]))
        ## Make sure the variance function can go to zero
        std_fn = (
            tf.slice(dist_params, [0, action_size], [batch_size, action_size]) + 1
        ) * 0.5
        std_lb = tf.ones(std_fn.shape) * self.std_lb  # set lower bound
        std_fn = tf.maximum(std_fn, std_lb)

        if self.const_std == "True":
            std_fn = tf.ones(std_fn.shape) * self.std_lb  # fix the standard deviation

        # factor = tf.math.pow(self.rate,tf.divide(action_count,self.decay_steps))
        # std_fn = tf.multiply(std_fn,tf.cast(factor,dtype=tf.float32))

        mean_fn, std_fn = self.apply_safety_layer(obs_ph, mean_fn, std_fn, cons_bounds)
        logp_action = self.gaussian_log_p(action_ph, mean_fn, std_fn)
        entropy = self.gaussian_entropy(std_fn)
        if not return_mean:
            return logp_action, entropy, cell_state
        else:
            return mean_fn, std_fn, logp_action, entropy, cell_state

    @tf.function
    def gaussian_entropy(self, std_fn):
        """the entropy of gaussian distribution"""
        k = std_fn.shape[1]
        det_sigma = tf.reduce_prod(std_fn, axis=1)
        return 0.5 * tf.math.log(tf.pow(2 * np.pi * np.e, k) * det_sigma)

    @tf.function
    def gaussian_log_p(self, action_ph, means, stds):
        """the log-likelihood function of gaussian distribution"""
        return (
            -0.5 * tf.reduce_sum(tf.square((action_ph - means) / stds), axis=1)
            - 0.5
            * np.log(2 * np.pi)
            * tf.cast(tf.shape(action_ph)[1], dtype=tf.float32)
            - tf.reduce_sum(tf.math.log(stds), axis=1)
        )

    @tf.function
    def safety_layer(self, obs_ph, mean_fn, std_fn, cons_bounds):
        N = obs_ph.shape[0]
        T = obs_ph.shape[1]
        states = tf.reshape(obs_ph, shape=(N * T, obs_ph.shape[2]))
        states = tf.slice(states, [0, 0], [N * T, self.state_dim])

        c_s = self.dynamics.current_cost(states)
        g_u = tf.matmul(self.dynamics.g_s(), tf.transpose(mean_fn))

        lambda_i = g_u + c_s - tf.cast(cons_bounds, dtype=tf.float32)
        # lambda_i = g_u + c_s - tf.cast(tf.reshape(cons_bounds, shape=(2,-1)),dtype=tf.float32)
        norm = tf.matmul(self.dynamics.g_s(), tf.transpose(self.dynamics.g_s()))
        norm = tf.reshape(tf.linalg.diag_part(norm), shape=(-1, 1))
        lambda_i = tf.nn.relu(lambda_i / norm)
        i = tf.math.argmax(lambda_i, axis=0)
        lambda_i = tf.reduce_max(lambda_i, axis=0)
        g_s = self.dynamics.g_s()
        safe_mean_fn = mean_fn - tf.matmul(
            tf.linalg.diag(lambda_i), tf.repeat(g_s, N * T, axis=0)
        )
        return tf.nn.tanh(safe_mean_fn), std_fn

    @tf.function
    def stochastic_safety_layer(self, obs_ph, mean_fn, std_fn, cons_bounds):
        Beta = 1.0
        N = obs_ph.shape[0]
        T = obs_ph.shape[1]
        states = tf.reshape(obs_ph, shape=(N * T, obs_ph.shape[2]))
        states = tf.slice(states, [0, 0], [N * T, self.state_dim])
        c_s = self.dynamics.current_cost(states)
        g_u_plus = tf.matmul(self.dynamics.g_s(), tf.transpose(mean_fn + Beta * std_fn))
        lambda_plus = g_u_plus + c_s - tf.cast(cons_bounds, dtype=tf.float32)
        g_u_minus = tf.matmul(
            self.dynamics.g_s(), tf.transpose(mean_fn - Beta * std_fn)
        )
        lambda_minus = g_u_minus + c_s - tf.cast(cons_bounds, dtype=tf.float32)
        norm = tf.matmul(self.dynamics.g_s(), tf.transpose(self.dynamics.g_s()))
        norm = tf.reshape(tf.linalg.diag_part(norm), shape=(-1, 1)) * (1 + Beta)
        lambda_plus = tf.nn.relu(lambda_plus / norm)
        lambda_minus = tf.nn.relu(lambda_minus / norm)
        lambda_plus = tf.reduce_max(lambda_plus, axis=0)
        lambda_minus = tf.reduce_max(lambda_minus, axis=0)
        g_s = self.dynamics.g_s()
        safe_mean_fn = mean_fn - tf.matmul(
            tf.linalg.diag(lambda_plus + lambda_minus), tf.repeat(g_s, N * T, axis=0)
        )
        safe_std_fn = std_fn + tf.matmul(
            tf.linalg.diag(-lambda_plus + lambda_minus), tf.repeat(g_s, N * T, axis=0)
        )
        # condition = tf.reshape(tf.multiply(lambda_plus,lambda_minus)>0, shape=(-1,1))
        # condition = tf.concat((condition,condition),axis=1)
        # safe_std_fn = tf.where(condition, std_fn, safe_std_fn)
        return tf.nn.tanh(safe_mean_fn), tf.clip_by_value(safe_std_fn, 1e-5, 10)

    @tf.function
    def identity_layer(self, obs_ph, mean_fn, std_fn, cons_bounds):
        return mean_fn, std_fn
