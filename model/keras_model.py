import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
from sklearn.metrics import confusion_matrix


def get_data():
    x_train = np.random.random((1000, 50))
    y_train = np.random.choice([0, 1], size=(1000,))
    x_test = np.random.random((200, 50))
    y_test = np.random.choice([0, 1], size=(200,))

    return x_train, y_train, x_test, y_test


def create_model():
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 50-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=50))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=10)
    score = model.evaluate(x_test, y_test, batch_size=10)

    return score


def predict_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    return cm
