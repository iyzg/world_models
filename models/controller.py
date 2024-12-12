import numpy as np

class Controller():
    def __init__(self, data, input_size, act_size):
        # TODO: Split into w/b
        self.input_size = input_size
        self.act_size = act_size
        self.w = data[:input_size * act_size].reshape(input_size, act_size)
        self.b = data[input_size * act_size:]

    def sigmoid(self, x):
        return np.clip(0.5 * (1 + np.tanh(x / 2)), 0, 1)

    def forward(self, x):
        out = np.dot(x, self.w) + self.b
        out = np.nan_to_num(out, nan=0.0)
        out[0] = np.tanh(out[0])
        out[1] = self.sigmoid(out[1])
        out[2] = self.sigmoid(out[2])
        return out
