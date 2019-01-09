from neuralnetwork.NeuralNetwork import NeuralNetwork
from classification.Classification import Classification
from classification.MongodbConnector import ia_data, insert_in_db
from random import choice
from datetime import datetime


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

data = []
#     {"class": "greeting", "sentence": "Hello comrade !"},
#     {"class": "greeting", "sentence": "How are you today ?"},
#     {"class": "greeting", "sentence": "Hello there"},
#     {"class": "greeting", "sentence": "Good morning fellows"},
#     {"class": "goodbye", "sentence": "Goodbye amigos"},
#     {"class": "goodbye", "sentence": "Let 's go now"},
#     {"class": "weather", "sentence": "What is the weather like today ?"},
#     {"class": "weather", "sentence": "It is quite sunny out there"},
#     {"class": "weather", "sentence": "It is raining today"},
#     {"class": "food", "sentence": "I want to eat potatoes"},
#     {"class": "food", "sentence": "I am so hungry"},
#     {"class": "food", "sentence": "What a delicious meal"},
#     {"class": "music", "sentence": "What song is this ?"},
#     {"class": "music", "sentence": "the song is beautiful"},
#     {"class": "music", "sentence": "Which artist sings this ?"},
#     {"class": "identity", "sentence": "What is your name ?"},
#     {"class": "identity", "sentence": "How are your called ?"},
#     {"class": "identity", "sentence": "Your name, please"},
#     {"class": "identity", "sentence": "name"}
# ]

for item in ia_data.find():
    data.append(item)

c = Classification(data)

out = c.generate_output()


start_timestamp = datetime.now().timestamp()

brain = NeuralNetwork(len(c.words), 30, len(c.classes))
brain.learning_rate = 0.25

iterations = 1000
for i in range(iterations):
    outputs = choice(out[0])
    inputs = out[1][out[0].index(outputs)]
    print(inputs)
    brain.train(inputs, outputs)

end_timestamp = datetime.now().timestamp()

time_delta = end_timestamp - start_timestamp

question = "Quelle heure il est ?"

prediction = brain.feed_forward(c.to_wordbag(question))

print(prediction)

print("This probably is a sentence of type:", c.classes[prediction.index(max(prediction))])

print(choice(data[prediction.index(max(prediction))]["responses"]))

# ------------------------------ Logs ----------------------------- #
log_file = open("logs.txt", "a")
log_file.write("\n[" + str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)")) + "] Test done with " + str(brain.nInputs) + " inputs, " + str(brain.nHidden) + " hidden neurons, " + str(brain.nOutputs) + " outputs and a learning rate of " + str(brain.learning_rate) + ". " + str(iterations) + " iterations performed over " + str(time_delta) + " seconds.")
log_file.write("\n      > results: " + c.classes[prediction.index(max(prediction))] + ", asking \"" + question + "\", raw values: " + str(prediction))
log_file.close()


# while True:
#     text = input("-> ")
#
#     prediction = brain.feed_forward(c.to_wordbag(text))
#
#     print("=> ",choice(data[prediction.index(max(prediction))]["responses"]))
#
#     print("\nC'était la bonne prédiction ? si oui, continue ! Sinon, tu t'attendais à quoi ?")
#     [print((c.classes.index(tag) + 1), "--", tag) for tag in c.classes]
#     print("7 -- c'était ok\n")
#     print("8 -- Arrêter\n")
#     wanted_result = input("> ")
#
#     stay = True
#     while(stay):
#         if int(wanted_result) > 0 and int(wanted_result) <= len(c.classes):
#             insert_in_db(c.classes[int(wanted_result) - 1], text)
#             print("\nMerci pour ton retour !\n\n")
#             stay = False
#         elif int(wanted_result) == 7:
#             stay = False
#             continue
#         elif int(wanted_result) == 8:
#             quit(0)
#         else:
#             print("what, réfléchis un peu")
