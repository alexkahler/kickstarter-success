from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from keras.layers import LeakyReLU
import pandas as pd
import numpy as np

## Show distribution and size of chosen categories
def get_distribution():
    Xs = pd.read_csv('kickstarter_preprocessed_final.csv')

    games = Xs.loc[Xs['category_games'] == 1]
    tech = Xs.loc[Xs['category_technology'] == 1]
    food = Xs.loc[Xs['category_food'] == 1]

    print("Games: " + str(len(games.index))+ '\n' + str(games['state'].value_counts(normalize=True)))
    print("tech: "+ str(len(tech.index))+ '\n'  + str(tech['state'].value_counts(normalize=True)))
    print("food: "+ str(len(food.index))+ '\n'  + str(food['state'].value_counts(normalize=True)))

    return

## Function for getting all data
def get_data():
    X = pd.read_csv('kickstarter_preprocessed_final.csv')
    scaler = StandardScaler()
    Y = X['state']
    print(Y.value_counts())
    X = X.drop(['state'], axis=1)
    X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

    return X, Y

## Function for getting slim data subset
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

    return X, Y

## Function for balancing and retrieving category based data, in the slim subset. Excluding features is possible
def get_balanced_slim_cat(category, exclusion = ''):
    scaler = StandardScaler()
    Xs = pd.read_csv('kickstarter_preprocessed_final.csv')

    cat = Xs.loc[Xs[category] == 1]
    cat0 = cat.loc[cat['state'] == 0]
    cat1 = cat.loc[cat['state'] == 1]

    if (len(cat0) == len(cat1)):
        Y = cat['state']
        X = cat[['blurb_length', 'usd_goal', 'creation_to_launch_hours', 'name_length', 'campaign_days', 'staff_pick_False', 'staff_pick_True']]

        if (exclusion != ''):
            X=X.drop(exclusion,axis=1)
            print(list(X))

        X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

        return X,Y

    elif(len(cat0) > len(cat1)):
        diff = len(cat0.index) - len(cat1.index)
        drop_indices = np.random.choice(cat0.index, diff, replace=False)
        cat0 = cat0.drop(drop_indices)
        print('sizes: ' + str(len(cat0.index)) + ' == ' + str(len(cat1)));
        X = pd.concat([cat0, cat1])
        X = X.sample(frac=1).reset_index(drop=True)
        Y = X['state']
        X = X[['blurb_length', 'usd_goal', 'creation_to_launch_hours', 'name_length', 'campaign_days', 'staff_pick_False', 'staff_pick_True']]

        if (exclusion != ''):
            print(list(X))
            X=X.drop(exclusion,axis=1)

        X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

        return X,Y

    else:
        diff = len(cat1.index) - len(cat0.index) #
        drop_indices = np.random.choice(cat1.index, diff, replace=False) #
        cat1 = cat1.drop(drop_indices)
        print('sizes: ' + str(len(cat1.index)) + ' == ' + str(len(cat0)));
        X = pd.concat([cat1, cat0])
        X = X.sample(frac=1).reset_index(drop=True)
        print(str(len(X.index)) + ' == ' + str(len(cat0) + len(cat1)))
        Y = X['state']
        X = X[
            ['blurb_length', 'usd_goal', 'creation_to_launch_hours', 'name_length', 'campaign_days',
             'staff_pick_False', 'staff_pick_True']]

        if (exclusion != ''):
            X=X.drop(exclusion,axis=1)
            print(list(X))

        X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

        return X,Y

def create_simple_model(input_dim = 10):
    model = Sequential()
    model.add(Dense(32, kernel_initializer='normal', activation='relu', input_dim=input_dim))
    model.add(Dropout(0.4))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_alt_model(input_dim = 10):
    model = Sequential()
    model.add(Dense(512, kernel_initializer='normal', input_dim=input_dim))
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

def create_cat_model(input_dim = 10):
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal', activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

## Function fo running k-folds cross-validation
def cross_validation(X, Y, n_splits=10, model='simple', epochs=100, batch_size=64):

    if model == 'alt':
        estimators = [('mlp', KerasClassifier(build_fn=create_alt_model, input_dim = len(X.columns), epochs=epochs, batch_size=batch_size, verbose=1))]
    elif model == 'cat':
        estimators = [('mlp', KerasClassifier(build_fn=create_cat_model, input_dim = len(X.columns), epochs=epochs, batch_size=batch_size, verbose=1))]
    else:
        estimators = [('mlp', KerasClassifier(build_fn=create_simple_model, input_dim = len(X.columns),epochs=epochs, batch_size=batch_size, verbose=1))]

    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    results = cross_validate(pipeline, X, Y, cv=kfold, scoring=['accuracy', 'neg_log_loss'])

    print('Accuracy: %.4f (%.2f%%)' % (results['test_accuracy'].mean(), results['test_accuracy'].std() * 100))
    print('Log loss: %.4f (%.2f%%)' % (results['test_neg_log_loss'].mean(), results['test_neg_log_loss'].std() * 100))

## Function for baseline prediction accuracy
def dummy_cross_validation(X, Y, n_splits=10):

    estimators = [('mlp', DummyClassifier(strategy="stratified"))]
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    results = cross_validate(pipeline, X, Y, cv=kfold, scoring=['accuracy', 'neg_log_loss'])

    print('Accuracy: %.4f (%.2f%%)' % (results['test_accuracy'].mean(), results['test_accuracy'].std() * 100))