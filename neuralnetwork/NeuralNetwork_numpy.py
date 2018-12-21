import numpy as np
from random import randrange


class NN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, nb_of_layers=1):
        # Params
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.nb_of_layers = nb_of_layers

        self.learning_rate = 0.1

        # Weights
        self.weights_ih = np.zeros((self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.zeros((self.output_nodes, self.hidden_nodes))

        self.randomize(self.weights_ih)
        self.randomize(self.weights_ho)

        # Biases
        self.bias_h = np.zeros((self.hidden_nodes, 1))
        self.bias_o = np.zeros((self.output_nodes, 1))

        self.randomize(self.bias_h)
        self.randomize(self.bias_o)

    def forward(self, inputs):
        hidden = np.multiply(self.weights_ih, inputs)
        np.add(hidden, self.bias_h)
        self.map(hidden, self.sigmoid)

        output = np.dot(self.weights_ho, hidden)
        np.add(output, self.bias_o)
        self.map(output, self.sigmoid)

        out = []
        for i in range(len(output)):
            for j in range(len(output[i])):
                out.append(output[i][j])

        return out

    def backward(self, output, inputs, targets):
        hidden = np.dot(self.weights_ih, inputs)

        # Calculate errors
        # E = targets - outputs
        output_error = np.subtract(targets - output)
        hidden_error = np.dot(np.transpose(self.weights_ho), output_error)

        self.map(output, self.sigmoid_derivative)
        np.multiply(output, output_error)
        np.multiply(output, self.learning_rate)

        weights_ho_deltas = np.dot(output, np.transpose(hidden))

        # Adjust output weights and biases
        np.add(self.weights_ho, weights_ho_deltas)
        np.add(self.bias_o, output)

        self.map(hidden, self.sigmoid_derivative)
        np.multiply(hidden, hidden_error)
        np.multiply(hidden, self.learning_rate)

        weights_ih_deltas = np.dot(hidden, np.transpose(inputs))

        # Adjust output weights and biases
        np.add(self.weights_ih, weights_ih_deltas)
        np.add(self.bias_h, hidden)

    def train(self, inputs, targets):
        out = self.forward(inputs)
        self.backward(out, inputs, targets)

    @staticmethod
    def randomize(array):
        for i in range(len(array)):
            for j in range(len(array[i])):
                array[i][j] = randrange(0, 9)

    @staticmethod
    def map(array, func):
        for i in range(len(array)):
            for j in range(len(array[i])):
                array[i][j] = func(array[i][j])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) / (1 - self.sigmoid(x))
