from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np


input_dim = 0

def get_data():
    x_train = np.random.random((1000, 50))
    y_train = np.random.choice([0, 1], size=(1000,))
    x_test = np.random.random((200, 50))
    y_test = np.random.choice([0, 1], size=(200,))

    return x_train, y_train, x_test, y_test


def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='normal', input_dim=50))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, epochs=50, batch_size=25):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return model


def evaluate_model(model, x_test, y_test, batch_size=25):
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    return score


def predict_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    return confusion_matrix(y_test, y_pred)


def cross_validation(X, Y, n_splits=10):
    print('Running Cross-Valiation')
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=25, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print('Results: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
