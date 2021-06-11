import json

import matplotlib.pyplot as plt
import numpy


class Features:
    def __init__(self):
        self.limit = 100
        self.X = []
        self.Y = []
        self.load_features(self.limit)

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def initialize(self):
        self.load_features(self.limit)

    def load_features(self, limit):
        with open("data/processed/dataset.json") as f:
            data = json.load(f)

            methods_labeled = data["methodsLabeled"]

            for method_index in methods_labeled:
                method_blocks = methods_labeled[method_index]["blocks"]
                for block in method_blocks:
                    tokens = method_blocks[block]["tokens"]
                    if len(tokens) <= limit:
                        self.X.append(method_blocks[block]["tokens"])
                        self.Y.append(method_blocks[block]["is_logged"])

    def convert_to_numpy(self):
        self.X = numpy.array([numpy.array(feature) for feature in self.X])
        self.Y = numpy.array([numpy.array(label) for label in self.Y])

    def find_max_size(self):
        max_size = 0
        for tokens in self.X:
            tokens_size = tokens.size
            if tokens_size > max_size:
                max_size = tokens_size
        return max_size

    def pad_features(self):
        max_size = self.find_max_size()
        self.X = numpy.array(
            [
                numpy.pad(
                    sequence,
                    (0, max_size - sequence.size),
                    "constant",
                    constant_values=("", ""),
                )
                for sequence in self.X
            ]
        )
        print(self.X.shape)

    def convert_to_numbers(self, key_to_index):
        map_function = numpy.vectorize(lambda token: key_to_index[token])

        self.X = map_function(self.X)

    def false_percentage(self):
        labels = self.getY()
        sum = 0
        for label in labels:
            if label == False:
                sum += 1
        return sum / len(labels)

    def show_length_histogram(self):
        lengths = [len(sequence) for sequence in loader.getX()]

        plt.hist(lengths, bins=30)
        plt.xlabel("Length")
        plt.ylabel("Number of sequences")
        plt.show()
