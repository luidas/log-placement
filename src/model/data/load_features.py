"""module that contains a class
"""
import json

import matplotlib.pyplot as plt
import numpy


class Features:
    """class to interact with the data"""

    def __init__(self):
        self.limit = 100
        self.features = []
        self.labels = []
        self.load_features(self.limit)

    def get_features(self):
        """returns the features

        Returns:
            [[string]]: list of lists of tokens
        """
        return self.features

    def get_labels(self):
        """gets the labels

        Returns:
            [bool]: list of labels
        """
        return self.labels

    def load_features(self, limit):
        """opens dataset stored at data/processed/dataset.json. reads values to a variable

        Args:
            limit (int): don't read sequences longer than this
        """
        with open("data/processed/dataset.json") as dataset:
            data = json.load(dataset)

            methods_labeled = data["methodsLabeled"]

            for method_index in methods_labeled:
                method_blocks = methods_labeled[method_index]["blocks"]
                for block in method_blocks:
                    tokens = method_blocks[block]["tokens"]
                    if len(tokens) <= limit:
                        self.features.append(method_blocks[block]["tokens"])
                        self.labels.append(method_blocks[block]["is_logged"])

    def convert_to_numpy(self):
        """converts lists to numpy arrays"""
        self.features = numpy.array([numpy.array(feature) for feature in self.features])
        self.labels = numpy.array([numpy.array(label) for label in self.labels])

    def find_max_size(self):
        """finds the maximum size of a feature sequence

        Returns:
            int: max size
        """
        max_size = 0
        for tokens in self.features:
            tokens_size = tokens.size
            if tokens_size > max_size:
                max_size = tokens_size
        return max_size

    def pad_features(self):
        """pads feature sequences with empty strings"""
        max_size = self.find_max_size()
        self.features = numpy.array(
            [
                numpy.pad(
                    sequence,
                    (0, max_size - sequence.size),
                    "constant",
                    constant_values=("", ""),
                )
                for sequence in self.features
            ]
        )

    @classmethod
    def convert_to_numbers(cls, key_to_index, array):
        """converts string to numbers

        Args:
            key_to_index (dict): dictionary to look up words integer index
            array (array of strings): array to convert

        Returns:
            array of ints: converted array
        """
        map_function = numpy.vectorize(lambda token: key_to_index[token])

        return map_function(array)

    def false_percentage(self):
        """calculate percentage of false labels

        Returns:
            int: percentage of false labels
        """
        labels = self.get_labels()
        false_labels = 0
        for label in labels:
            if label is False:
                false_labels += 1
        return sum / len(labels)

    def show_length_histogram(self):
        """show histogram of feature lengths"""
        lengths = [len(sequence) for sequence in self.get_features()]

        plt.hist(lengths, bins=30)
        plt.xlabel("Length")
        plt.ylabel("Number of sequences")
        plt.show()
