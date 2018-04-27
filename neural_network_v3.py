import numpy as np


def fit():
    ...

class Web():
    def __init__(self, size, training_rate):
        ...


class Layer():
    def __init__(self, size, weights=0, bias=0, bias_weights=0):
        if len(size) < 2:
            self.deltas = np.zeros(size[0])
            self.outs = np.zeros(size[0])
        else:
            self.weights = np.random.rand(size[0], size[1])
            self.corrections = np.zeros(size[0], size[1])
            self.deltas = np.zeros(size[0])
            self.outs = np.zeros(size[0])

    def derivative(self, x):
        return np.multiply(1.0 - x, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def delta(self, weights, delta, deriv):
        return np.multiply(weights * delta.T, deriv.T)

    def change_corrections(self, delta, out, training_rate):
        return np.multiply(out, delta) * training_rate