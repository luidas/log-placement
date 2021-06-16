"""
processes data
creates word embedding layer
creates a neural network
fits network outputs graph with metrics
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential

from data.load_features import Features

features = Features()
features.convert_to_numpy()
features.pad_features()

X = features.getX()
Y = features.getY()


def get_embeddings(minimum, dimensions, window):
    """Method to get word embeddings

    Args:
        minimum (int): count threshold
        dimensions (int): size of word vectors
        window (int): distance between the current and predicted word

    Returns:
        KeyedVectors: mapping between words and embeddings
    """
    emb_model = Word2Vec(
        X.tolist(),
        min_count=minimum,
        vector_size=dimensions,
        workers=4,
        window=window,
        sg=1,
    )
    word_vectors = emb_model.wv
    return word_vectors


word2vec = get_embeddings(0, 200, 5)

X = features.convert_to_numbers(word2vec.key_to_index, X)


(x_train, y_train), (x_test, y_test) = (
    X[: int(X.shape[0] * 0.8)],
    Y[: int(Y.shape[0] * 0.8)],
), (
    X[int(X.shape[0] * 0.8) :],
    Y[int(Y.shape[0] * 0.8) :],
)


def gensim_to_keras_embedding(keyed_vectors, train_embeddings=False):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    # keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
    # index_to_key = (
    #     keyed_vectors.index_to_key
    # )  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer


model = Sequential()
model.add(gensim_to_keras_embedding(word2vec))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", Precision(), Recall()],
)

# keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

precisions_test = history.history["val_precision"]
recalls_test = history.history["val_recall"]

f_test = [(2 * p * r) / (p + r) for p in precisions_test for r in recalls_test]

plt.plot(f_test)
plt.plot(precisions_test)
plt.plot(recalls_test)
plt.ylabel("Metric")
plt.xlabel("Epoch")
plt.legend(["F-Measure", "Precision", "Recall"], loc="upper left")
plt.show()
