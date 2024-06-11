import numpy as np
import tensorflow as tf


class MinMaxScaler:
    def __init__(self, scale_min, scale_max) -> None:
        self.min = scale_min
        self.max = scale_max

    def fit(self, x):
        self.data_min = tf.reduce_min(x)
        self.data_max = tf.reduce_max(x)
        self.diff = self.data_max - self.data_min

    def transform(self, x):
        return (self.max - self.min) * (x - self.data_min) / self.diff + self.min

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        out = (x - self.min) / (self.max - self.min) * self.diff + self.data_min
        return out.numpy()


if __name__ == "__main__":
    scaler = MinMaxScaler(-1, 1)
    data = np.random.rand(2, 2) * 10
    print("original data\n", data)
    d = scaler.fit_transform(data)
    print("scalered data\n", d)
    inv_d = scaler.inverse_transform(d)
    print("inverse transformed \n", inv_d)
