import numpy as np


class Neuro_web():
    """
    A class that stores layers of our network and produces direct and reverse propagation
    """
    def __init__(self, training_rate, last_layer_size, bias=0):
        """
        Initialize neuro web and creating output layer

        Arguments:
            training_rate - Training speed of neuro web
            last_layer_size - Number of neurons in the layer

        Return:
            Initialize layers array and append outup layer
            Initialize training rate variable
            Initialize web size variable which store number of layer on web
        """
        self.layers = []
        last_layer = Layer(last_layer_size, 0, layer_type=1)
        self.layers.append(last_layer)
        self.training_rate = training_rate
        self.web_size = 1

    def new_layer(self, size, bias=0):
        """
        Add new neuro web layer consisted of Neuron objects

        Arguments:
            size - Number of neurons in the layer

        Return:
            append layer object in layers array
        """
        web_size = len(self.layers)
        prev_layer_size = len(self.layers[web_size - 1].neurons)
        layer = Layer(size, prev_layer_size, layer_type=0)
        self.layers.append(layer)
        self.web_size += 1

    def train_network(self, input_data, epoch):
        """
        Function realizing direct and reverse propagation through a neural network with a given number of epochs

        Arguments:
            input_data - Input training data
            epoch - Number of training epoch
        Return:
            Each iteration of the epoch cycle returns a new neural network with the corrected weights coefficients
        """
        def print_network():
            print()
            for x in range(len(self.layers)):
                print(f"Layer {x}")
                if x == self.web_size - 1:
                    print(f"Delta of neuron -----> {self.layers[x].neurons[0].delta}")
                else:
                    for i in range(len(self.layers[x].neurons)):
                        print(f"Weights of neuron -----> {self.layers[x].neurons[i].weights}")
                        print(f"Delta of neuron -----> {self.layers[x].neurons[i].delta}")

        self.layers.reverse()
        print_network()
        for x in range(epoch):
            np.random.shuffle(input_data)
            print(f"Current epoch {x}")
            for data in input_data:
                print(f"Input data {data[0][0]}, {data[0][1]}")
                for inp in range(len(self.layers[0].neurons)):
                    self.layers[0].neurons[inp].out = data[0][inp]

                for layer in range(1, self.web_size):
                    self.layers[layer].direct_distr(self.layers[layer - 1])

                for layer in range (self.web_size - 1, -1, -1):
                    if layer == self.web_size - 1:
                        self.layers[layer].back_grad_calc(data[1])
                    else:
                        self.layers[layer].back_prop_hidden(self.layers[layer + 1], self.training_rate)
            if x < 2:
                print_network()

class Layer():
    def __init__(self, size, prev_layer_size, bias=0, layer_type=0):
        self.neurons = []
        self.layer_size = size
        self.prev_layer_size = prev_layer_size
        self.biases = bias
        if layer_type == 0:
            for x in range(size):
                x = Neuron(prev_layer_size)
                self.neurons.append(x)
        else:
            for x in range(size):
                x = Last_layer_neuron()
                self.neurons.append(x)

    def direct_distr(self, prev_layer):
        for neuron in range(self.layer_size):
            weights = []
            outs = []
            for neu in prev_layer.neurons:
                outs.append(neu.out)
                weights.append(neu.weights[neuron])
            self.neurons[neuron].sigmoid_func(weights, outs, self.biases)

    def back_prop_hidden(self, prev_layer, training_rate):
        for neuron in range(self.layer_size):
            deriv = self.neurons[neuron].derivative()
            prev_deltas = []
            for neu in prev_layer.neurons:
                prev_deltas.append(neu.delta)
            #print(prev_deltas)
            self.neurons[neuron].hidden_delta(prev_deltas, deriv)

            for neu in range(len(prev_layer.neurons)):
                prev_delta = prev_layer.neurons[neu].delta
                self.neurons[neuron].grad_change(neu, prev_delta, training_rate)

    def back_grad_calc(self, ideal):
        result_error = 0
        i = 0
        for neuron in self.neurons:
            deriv = neuron.derivative()
            neuron.output_delta(ideal[i], deriv)
            error = neuron.mse_error(ideal[i])
            print(f"Результат работы {i}-го выходного нейрона {self.neurons[i].out}")
            if (ideal - self.neurons[i].out)**2 < 0.25:
                print("Правильно")
            else:
                print("Не правильно")
            #print(f"Ошибка {i} выходного нейрона равна ---> {error}")
            result_error += error
            i += 1
        print(f"Результирующая ошибка равна ---> {result_error}")


class Neuron():
    def __init__(self, prev_layer_size, **kwargs):
        if "weights" in kwargs:
            self.weights = kwargs["weights"]
        else:
            self.weights = np.random.random(prev_layer_size)
        self.out = 0
        self.delta = 0
        self.change_value = 0

    # Inputs - weights - array with input neuron weight, outs - values of input neurons
    # Result - working result of sigmoid function
    def sigmoid_func(self, weights, outs, bias):
        size = len(weights)
        sumation_value = 0
        for communication in range(size):
            sumation_value += weights[communication] * outs[communication]
        sumation_value += bias
        self.out = 1 / (1 + np.exp(sumation_value))
        return self.out

    # Inputs - output value neuron
    # Result - deriviative of sigmoid function
    def derivative(self):
        return (1 - self.out ) * self.out

    def hidden_delta(self, prev_deltas, deriv):
        summation = 0
        size = len(self.weights)
        # print(f"Self weights and previous delta {self.weights}\n{prev_deltas}")
        for x in range(size):
            summation += self.weights[x] * prev_deltas[x]
        self.delta = summation * deriv

    # Calculating gradients and changing weights
    def grad_change(self, prev_neuron, prev_delta, training_rate):
        gradient = prev_delta * self.out
        delta_weight = gradient * training_rate
        self.weights[prev_neuron] += delta_weight / 2



class Last_layer_neuron():
    def __init__(self):
        self.out = 0
        self.delta = 0
        self.change_value = 0
        self.mse = 0

    # Inputs - weights - array with input neuron weight, outs - values of input neurons
    # Result - working result of sigmoid function
    def sigmoid_func(self, weights, outs, bias):
        size = len(weights)
        sumation_value = 0
        for communication in range(size):
            sumation_value += weights[communication] * outs[communication]
        sumation_value += bias
        self.out = 1 / (1 + np.exp(-sumation_value))
        return self.out

    def mse_error(self, ideal):
        self.mse = (ideal - self.out) ** 2
        return self.mse

    # Inputs - output value neuron
    # Result - deriviative of sigmoid function
    def derivative(self):
        return (1 - self.out) * self.out

    def output_delta(self, ideal, deriv):
        self.delta = (ideal - self.out)


np.random.seed(75)

DATA = [
    ([0, 0], [0, ]),
    ([0, 1], [1, ]),
    ([1, 0], [1, ]),
    ([1, 1], [0, ])
]

#DATA = [
#    ([0, 0, 0], 0),
#    ([1, 0, 0], 1),
#    ([0, 1, 0], 0),
#    ([1, 0, 1], 1),
#    ([0, 1, 1], 1),
#    ([1, 1, 0], 0),
#    ([1, 1, 1], 1),
#    ([0, 0, 1], 1)
#]

neuro_web = Neuro_web(0.7, 1)
neuro_web.new_layer(2, bias=1)
neuro_web.new_layer(2, bias=1)

inp_data = DATA.copy()
neuro_web.train_network(inp_data, 2000)
