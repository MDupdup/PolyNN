from neuralnetwork.NeuralNetwork import NeuralNetwork
from classification.Classification import Classification
from random import choice


# def normalize_ascii_value(n):
#     return (n - 32) / 90
#
#
# def format_string_to_array(string):
#     out = []
#     for char in list(string):
#         out.append(normalize_ascii_value(ord(char)))
#
#     while len(out) < 8:
#         out.append(normalize_ascii_value(32))
#
#     return out
#
#
# hello = "hello!"
# hello_array = []
#
# bye = "Goodbye"
# bye_array = []
#
# data = [
#     {
#         "inputs": format_string_to_array(hello),
#         "targets": [1]
#     },
#     {
#         "inputs": format_string_to_array(bye),
#         "targets": [0]
#     }
# ]
#
# print("is this a hello?", brain.feed_forward(format_string_to_array("hello!")))
# print("is this a goodbye?", brain.feed_forward(format_string_to_array("Goodbye")))
# print("is this a random word?", brain.feed_forward(format_string_to_array("kluguk")))

data = [
    {"class": "greeting", "sentence": "Hello comrade !"},
    {"class": "greeting", "sentence": "How are you today ?"},
    {"class": "greeting", "sentence": "Hello there"},
    {"class": "greeting", "sentence": "Good morning fellows"},
    {"class": "goodbye", "sentence": "Goodbye amigos"},
    {"class": "goodbye", "sentence": "Let 's go now"},
    {"class": "weather", "sentence": "What is the weather like today ?"},
    {"class": "weather", "sentence": "It is quite sunny out there"},
    {"class": "weather", "sentence": "It is raining today"},
    {"class": "food", "sentence": "I want to eat potatoes"},
    {"class": "food", "sentence": "I am so hungry"},
    {"class": "food", "sentence": "What a delicious meal"},
    {"class": "music", "sentence": "What song is this ?"},
    {"class": "music", "sentence": "the song is beautiful"},
    {"class": "music", "sentence": "Which artist sings this ?"}
]

c = Classification(data)

out = c.generate_output()

print(out[0][0])
print(out[1][0])

brain = NeuralNetwork(len(c.words), int((len(c.words)+len(c.classes))/2), len(c.classes))
brain.learning_rate = 0.2

for i in range(20000):
    outputs = choice(out[0])
    inputs = out[1][out[0].index(outputs)]
    print(outputs)
    print(inputs)
    brain.train(inputs, outputs)

prediction = brain.feed_forward(c.to_wordbag("Hello, is it raining today ?"))

print(prediction)

print("This probably is a sentence of type:", c.classes[prediction.index(max(prediction))])
