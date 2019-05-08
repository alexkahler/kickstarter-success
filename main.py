from model.keras_model import *


def main():

    x_train, y_train, x_test, y_test = get_data()
    model = create_model()
    result = train_model(model, x_train, y_train, x_test, y_test)

    print('Loss & Accuracy: ', result)


if __name__ == "__main__":
    main()
