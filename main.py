from model.keras_model import *
import numpy as np
import sklearn


def main():
    # Get data
    # x_train, y_train, x_test, y_test = get_data()
    x_train, y_train, x_test, y_test = get_slim_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    print(sorted(sklearn.metrics.SCORERS.keys()))
    # Create and train model
    # model = create_model()
    # trained_model = train_model(model, x_train, y_train, 150, 256)

    # Evaluate model
    # result = evaluate_model(trained_model, x_test, y_test, 128)
    # cm = predict_model(trained_model, x_test, y_test)
    # print('Loss & Accuracy: ', result)
    # print('Confusion Matrix:', cm)

    cross_validation(X, Y, 2)


if __name__ == "__main__":
    main()
