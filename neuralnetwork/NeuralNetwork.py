import numpy as np
from Matrix import Matrix


class NeuralNetwork(object):
    def __init__(self, nInputs, nHidden, nOutputs):
        # learning rate
        self.learning_rate = 0.1

        # nodes
        self.nInputs = nInputs
        self.nHidden = nHidden
        self.nOutputs = nOutputs

        #self.nHidden = int(15 / (2 * (self.nInputs + self.nOutputs)))

        # weights
        self.weights_ih = Matrix(self.nHidden, self.nInputs)
        self.weights_ho = Matrix(self.nOutputs, self.nHidden)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        # biases
        self.bias_h = Matrix(self.nHidden, 1)
        self.bias_o = Matrix(self.nOutputs, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

    # Pass inputs through the network
    def feed_forward(self, array):
        inputs = Matrix.from_array(array)

        # Generate hidden outputs
        hidden = Matrix.dot(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden = Matrix.map(hidden, self.sigmoid)

        # Generate output
        output = Matrix.dot(self.weights_ho, hidden)
        output.add(self.bias_o)
        output = Matrix.map(output, self.sigmoid)

        return output.to_array()

    def train(self, inputs_array, targets_array):
        inputs = Matrix.from_array(inputs_array)

        # Generate hidden outputs
        hidden = Matrix.dot(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden = Matrix.map(hidden, self.sigmoid)

        # Generate output
        outputs = Matrix.dot(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs = Matrix.map(outputs, self.sigmoid)

        targets = Matrix.from_array(targets_array)

        # Calculate output errors
        # e = targets - outputs
        targets.show()
        outputs.show()
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate hidden layer errors
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.dot(weights_ho_t, output_errors)

        # Calculate output deltas
        gradients = Matrix.map(outputs, self.dsigmoid)
        gradients.multiply(output_errors)
        gradients.multiply(self.learning_rate)

        hidden_T = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.dot(gradients, hidden_T)

        # Adjust output weights&bias
        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradients)

        # Calculate hidden deltas
        hidden_gradient = Matrix.map(hidden, self.dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)

        inputs_T = Matrix.transpose(inputs)

        weights_ih_deltas = Matrix.dot(hidden_gradient, inputs_T)

        # Adjust hidden weights&bias
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)

    # Activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Activation function derivative
    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)
