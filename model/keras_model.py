from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from keras.layers import LeakyReLU
import pandas as pd


def get_data():
    X = pd.read_csv('kickstarter_preprocessed_final.csv')
    scaler = StandardScaler()
    Y = X['state']
    X = X.drop(['state'], axis=1)
    X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    return X, Y


def get_slim_data():
    X = pd.read_csv('kickstarter_preprocessed_final.csv')
    scaler = StandardScaler()
    Y = X['state']
    X = X.drop(['state'], axis=1)
    X = X[['blurb_length', 'usd_goal', 'creation_to_launch_hours', 'name_length', 'campaign_days', 'staff_pick_False',
           'staff_pick_True', 'category_art', 'category_comics', 'category_crafts', 'category_dance', 'category_design',
           'category_fashion', 'category_film & video',
           'category_food', 'category_games', 'category_journalism', 'category_music', 'category_photography',
           'category_publishing', 'category_technology', 'category_theater']]
    X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    return X, Y


def create_simple_model():
    model = Sequential()
    model.add(Dense(32, kernel_initializer='normal', activation='relu', input_dim=83))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_optimized_model():
    model = Sequential()
    model.add(Dense(512, kernel_initializer='normal', input_dim=22))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.4))
    model.add(Dense(256, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.4))
    model.add(Dense(128, kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, epochs=25, batch_size=25):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return model


def evaluate_model(model, x_test, y_test, batch_size=25):
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    return score


def predict_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    return confusion_matrix(y_test, y_pred)


def cross_validation(X, Y, n_splits=10, model='simple', epochs=100):

    if model == 'simple':
        estimators = [('mlp', KerasClassifier(build_fn=create_simple_model, epochs=epochs, batch_size=256, verbose=1))]
    else:
        estimators = [('mlp', KerasClassifier(build_fn=create_optimized_model, epochs=epochs, batch_size=256, verbose=1))]

    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    results = cross_validate(pipeline, X, Y, cv=kfold, scoring=['accuracy', 'neg_log_loss'])

    print('Accuracy: %.4f (%.2f%%)' % (results['test_accuracy'].mean(), results['test_accuracy'].std() * 100))
    print('Log loss: %.4f (%.2f%%)' % (results['test_neg_log_loss'].mean(), results['test_neg_log_loss'].std() * 100))
