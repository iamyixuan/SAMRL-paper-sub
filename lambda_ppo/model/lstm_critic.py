import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(
        self,
        output_dim,
        repr_sizes=(32,),
        hidden_sizes=(32,),
        batch_size=50,
        h_size=64,
        activation=tf.nn.tanh,
        output_activation=None,
    ):
        super(Critic, self).__init__()

        self.base_model = []
        self.activation = activation
        self.output_activation = output_activation
        self.repr_sizes = repr_sizes
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        for h in self.repr_sizes:
            self.base_model.append(
                tf.keras.layers.Dense(
                    units=h,
                    activation=self.activation,
                    kernel_initializer=tf.keras.initializers.Orthogonal(
                        gain=tf.sqrt(2.0)
                    ),
                )
            )
        self.h_size = h_size
        self.batch_size = batch_size
        self.lstm1 = tf.keras.layers.LSTM(
            units=h_size,
            return_state=True,
            stateful=False,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )

        self.heads = []
        for _ in range(self.output_dim):
            head_mlp = []
            for h in self.hidden_sizes:
                head_mlp.append(
                    tf.keras.layers.Dense(
                        units=h,
                        activation=self.activation,
                        kernel_initializer=tf.keras.initializers.Orthogonal(
                            gain=tf.sqrt(2.0)
                        ),
                    )
                )
            head_mlp.append(
                tf.keras.layers.Dense(
                    units=1,
                    activation=self.output_activation,
                    kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
                )
            )
            self.heads.append(head_mlp)

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
        if lstm_state is None:
            self.lstm1.get_initial_state = self.get_initial_zero_state
        else:
            self.hidden_state, self.cell_state = lstm_state[0], lstm_state[1]
            self.lstm1.get_initial_state = self.get_initial_state

        x = tf.cast(x, tf.float32)
        x, hidden_state, cell_state = self.lstm1(x)
        N = x.shape[0]  # number of samples
        T = x.shape[1]  # number of timesteps
        x = tf.reshape(x, shape=(N * T, x.shape[2]))  # reshape to unroll output
        for layer in self.base_model:
            x = layer(x)
        output = []
        for i in range(self.output_dim):
            x_head = self.heads[i][0](x)
            for layer in self.heads[i][1:]:
                x_head = layer(x_head)
            output.append(x_head)
        return tf.stack(output, axis=1), (hidden_state, cell_state)
