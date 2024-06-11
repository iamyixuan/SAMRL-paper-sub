import tensorflow as tf
from joblib import load
import numpy as np
from .ghfr_gym_td import SINDyGYM
import matplotlib.pyplot as plt

model_path = ".../env_data/red_SINDYc_model_2022-4-15-downsample.joblib"


class sindyDx(tf.keras.Model):
    def __init__(self):
        super(sindyDx, self).__init__()

        self.n_states = 13
        self.n_inputs = 1

        self.min_u = -1.0
        self.max_u = 1.0
        # These are the SINDYc coefficients
        # ignores the first columns of constant values
        self.model = load(model_path)
        coeffs = self.model.coefficients()
        self.coefs = tf.cast(tf.constant(coeffs[:, 1:]), dtype=tf.float32)

    @tf.function
    def call(self, x, u):
        x = tf.cast(x, dtype=tf.float32)
        u = tf.cast(u, dtype=tf.float32)
        tf.debugging.assert_equal(tf.rank(x), 2)  # batch, x_dim
        tf.debugging.assert_equal(tf.rank(u), 2)  # batch, u_dim

        tf.debugging.assert_equal(x.shape[1], self.n_states)
        tf.debugging.assert_equal(u.shape[1], self.n_inputs)

        u = tf.clip_by_value(u, self.min_u, self.max_u)

        # s_x = tf.concat((x, u,tf.math.pow(x, 2), tf.math.pow(u, 2), tf.math.pow(x, 3),tf.math.pow(u, 3)),axis=1)
        s_x = tf.concat((x, u), axis=1)  # linear model
        return tf.transpose(tf.matmul(self.coefs, tf.transpose(s_x)))

    # @tf.function
    def gradient_wrt_a(self):
        # A[:,4:6]
        return self.coefs[:, self.n_states : self.n_states + self.n_inputs]

    # @tf.function
    def linear_dynamics_matrix(self):
        # A
        return self.coefs[:, 0 : self.n_states + self.n_inputs]

    # @tf.function
    def gradient_wrt_s(self):
        # A[:,0:4]
        return self.coefs[:, 0 : self.n_states]

    @tf.function
    def constraint_function(self):
        return tf.constant(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

    # @tf.function
    def current_cost(self, s):
        # s should be 1x 4
        s = tf.cast(s, dtype=tf.float32)
        return tf.matmul(
            tf.matmul(self.constraint_function(), self.gradient_wrt_s()),
            tf.transpose(s),
        )

    # @tf.function
    def g_s(self):
        return tf.matmul(self.constraint_function(), self.gradient_wrt_a())


if __name__ == "__main__":
    sindyNN = sindyDx()
    env = SINDyGYM()

    o = env.reset()
    print("the shape of the state is", o.shape)

    s = env.s_tr_ls[1][0].reshape(1, -1)
    a = env.a_tr_ls[1][0].reshape(1, -1)

    print("s shape is", s.shape)

    # const_coef = sindyNN.model.coefficients()[:,0]
    next_state_sindy = sindyNN.model.predict(x=s, u=a)
    print(next_state_sindy.shape)
    # next_state_tf = sindyNN(x=s,u=a)

    # A  = sindyNN.linear_dynamics_matrix()
    # x = np.concatenate((s,a),axis=1) #1x6
    # new_state_approx = np.matmul(A, x.T).T #1x4

    # new_cost = sindyNN.current_cost(s) + np.matmul(sindyNN.g_s().numpy(), a.T)

    # sindy_pred = []
    # tf_pred = []
    # N= env.s_tr_ls[1].shape[0]
    # s_id = 0
    # for i in range(N):
    #     s = env.s_tr_ls[1][i].reshape(1,-1)
    #     a = env.a_tr_ls[1][i].reshape(1,-1)
    #     next_state_sindy = sindyNN.model.predict(x=s,u=a)
    #     next_state_tf = sindyNN(x=s,u=a)
    #     sindy_pred.append(next_state_sindy.ravel()[s_id])
    #     tf_pred.append(next_state_tf.numpy().ravel()[s_id])
    # plt.plot(sindy_pred, label='SINDY')
    # plt.plot(tf_pred,label = 'NN')
    # #predictions will be off by the const_coeff
    # plt.legend()
    # plt.show()
