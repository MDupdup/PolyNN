from neuralnetwork.NeuralNetwork import NeuralNetwork
from neuralnetwork.NeuralNetwork_numpy import NN
from random import choice


def normalize_ascii_value(n):
    return (n - 32) / 90


def format_string_to_array(string):
    out = []
    for char in list(string):
        out.append(normalize_ascii_value(ord(char)))

    while len(out) < 8:
        out.append(normalize_ascii_value(32))

    return out


hello = "hello!"
hello_array = []

bye = "Goodbye"
bye_array = []

data = [
    {
        "inputs": format_string_to_array(hello),
        "targets": [1]
    },
    {
        "inputs": format_string_to_array(bye),
        "targets": [0]
    }
]

# np_brain = NN(8, 5, 1)
# np_brain.learning_rate = 0.1

brain = NeuralNetwork(8, 5, 1)
brain.learning_rate = 0.1

for i in range(50000):
    dataset = choice(data)
    brain.train(dataset["inputs"], dataset["targets"])

print("is this a hello?", brain.forward(format_string_to_array("hello!")))
print("is this a goodbye?", brain.forward(format_string_to_array("Goodbye")))
print("is this a random word?", brain.forward(format_string_to_array("kluguk")))
