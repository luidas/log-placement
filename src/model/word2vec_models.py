import json
import sys

import numpy
from gensim.models import Word2Vec

from data.load_features import Features


loader = Features()
loader.convert_to_numpy()
loader.pad_features()

X = loader.getX()

minimums = [0]
dimensionss = [200]
windows = [5]

for minimum in minimums:
    for dimensions in dimensionss:
        for window in windows:
            model = Word2Vec(X.tolist(), min_count=minimum, vector_size=dimensions, workers=4, window=window, sg=1)
            word_vectors = model.wv
            word_vectors.save("models/word2vec/unfiltered_unlimited.wordvectors")
