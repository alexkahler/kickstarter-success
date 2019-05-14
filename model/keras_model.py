from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def get_data():
    X = pd.read_csv('kickstarter_preprocessed_final.csv')

    scaler = StandardScaler()

    Y = X['state']
    X = X.drop(['state'], axis=1)

    X = pd.DataFrame(scaler.fit_transform((X)), columns=list(X.columns))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    return x_train, y_train, x_test, y_test


def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=83))
    # model.add(Dropout(0.4))
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
    estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=5, batch_size=5, verbose=1)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print('Results: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
