from model.keras_model import *


def main():
    # Get data
    x_train, y_train, x_test, y_test = get_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    # Create and train model
    model = create_model()
    trained_model = train_model(model, x_train, y_train, 50, 20)

    # Evaluate model
    result = evaluate_model(trained_model, x_test, y_test, 20)
    cm = predict_model(trained_model, x_test, y_test)
    print('Loss & Accuracy: ', result)
    print('Confusion Matrix:', cm)

    # cross_validation(X, Y, 2)


if __name__ == "__main__":
    main()
