import numpy as np


class Web():
    def __init__(self, size, training_rate):
        def debug():
            i = 0
            for x in self.layers:
                print(f"------------Layer {i}------------")
                if x == self.layers[self.web_size - 1]:
                    print("Deltas")
                    print(x.deltas)
                else:
                    print("Weights")
                    print(x.weights)
                    print("Deltas")
                    print(x.deltas)
                i += 1

        self.layers = []
        self.training_rate = training_rate
        self.web_size = len(size)
        for x in range(len(size)):
            layer_gen = size[x:x+2]
            layer = Layer(layer_gen)
            self.layers.append(layer)
        debug()

    def fit(self, data, epoch):
        def direct_prop(data_unit, ideal):
            self.layers[0].outs = data_unit
            for layer in range(1, self.web_size):
                self.layers[layer].direct_distr(self.layers[layer - 1])
                if layer == self.web_size - 1:
                    self.layers[layer].mse(ideal)

        def back_propagation(data_unit, ideal):
            for layer in reversed(range(self.web_size)):
                if layer == self.web_size - 1:
                    self.layers[layer].back_prop(0, ideal=ideal)
                else:
                    self.layers[layer].back_prop(self.layers[layer + 1])

            for correction in range(self.web_size - 1):
                self.layers[correction].weights_correction(len(data_unit), self.training_rate)

        def epoch_iter(data):
            working_data = data.copy()
            np.random.shuffle(working_data)
            for x, y in working_data:
                direct_prop(x, y)
                back_propagation(x, y)

        for ep in range(epoch):
            epoch_iter(data)


class Layer():
    def __init__(self, size, bias=0, bias_weights=0):
        if len(size) < 2:
            self.deltas = np.zeros(size[0])
            self.outs = np.zeros(size[0])
        else:
            self.weights = np.random.rand(size[0], size[1])
            self.corrections = np.zeros((size[0], size[1]))
            self.deltas = np.zeros(size[0])
            self.outs = np.zeros(size[0])

    def direct_distr(self, prev_layer):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        sigmoid_input = np.dot(prev_layer.weights.T, prev_layer.outs.T)
        self.outs = sigmoid(sigmoid_input.T)

    def back_prop(self, prev_layer, ideal=0):
        def delta(weights, delta, deriv):
            return np.multiply(weights * delta.T, deriv.T)

        def derivative(x):
            return np.multiply(1.0 - x, x)

        def calc_corrections(outputs, prev_deltas):
            return np.multiply(outputs.T, prev_deltas)

        if prev_layer == 0:
            self.deltas = self.outs - ideal
        else:
            deriv = derivative(self.outs)
            self.deltas = delta(self.weights, prev_layer.deltas, deriv)
            self.corrections += calc_corrections(self.outs.T, prev_layer.deltas)

    def mse(self, ideal):
        size = len(self.outs)
        print(f"Значения выходных нейронов {self.outs}")
        print(f"Общая ошибка сети ---> {np.sum(np.multiply(self.outs, ideal)) / size}")

    def weights_correction(self, data_size, training_rate):
        self.weights -= self.corrections / data_size * training_rate


DATA = [
       ([0, 0], [0, ]),
       ([0, 1], [1, ]),
       ([1, 0], [1, ]),
       ([1, 1], [0, ])
       ]

web = Web([2, 3, 2], 0.75)

