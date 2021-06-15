import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential

from data.load_features import Features


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


word2vec = KeyedVectors.load("models/word2vec/unfiltered_unlimited.wordvectors")

features = Features()

features.convert_to_numpy()
features.pad_features()
features.convert_to_numbers(word2vec.key_to_index)

X = features.getX()
Y = features.getY()

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

keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

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
