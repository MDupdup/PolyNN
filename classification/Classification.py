import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


class Classification(object):
    def __init__(self, training_data):
        self.words = []
        self.classes = []
        self.documents = []

        self.ignore_words = ["?", "!"]

        for pattern in training_data:
            words = nltk.word_tokenize(pattern['sentence'])
            self.words.extend(words)

            self.documents.append((words, pattern['class']))

            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        self.words = [stemmer.stem(word.lower()) for word in self.words if word not in self.ignore_words]

        self.words = list(set(self.words))
        self.classes = list(set(self.classes))

        print(self.words)


    def generate_output(self):
        words_bag = []
        output = []

        for doc in self.documents:
            bag = []
            output_empty = [0] * len(self.classes)

            pattern_words = doc[0]

            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)

            words_bag.append(bag)

            output_empty[self.classes.index(doc[1])] = 1
            output.append(output_empty)

        return [output, words_bag]


    def to_binary(self, sentence):
        bag = []

        sentence = nltk.word_tokenize(sentence)

        sentence = [stemmer.stem(word.lower()) for word in sentence]

        for word in self.words:
            if word in sentence:
                print("found this",word)
                bag.append(1)
            else: bag.append(0)

        return bag
