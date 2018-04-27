import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = sigmoid(2)

print(sigmoid(a))