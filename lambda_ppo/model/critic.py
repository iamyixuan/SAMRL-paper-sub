import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(
        self,
        output_dim,
        repr_sizes=(32,),
        hidden_sizes=(32,),
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
    def call(self, x):
        x = tf.cast(x, tf.float32)
        for layer in self.base_model:
            x = layer(x)
        output = []
        for i in range(self.output_dim):
            x_head = self.heads[i][0](x)
            for layer in self.heads[i][1:]:
                x_head = layer(x_head)
            output.append(x_head)
        return tf.stack(output, axis=1)
