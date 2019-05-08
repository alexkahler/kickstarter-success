from model.keras_model import *
import numpy as np
import keras


def main():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = create_model()
    result = train_model(model, x_train, y_train, x_test, y_test)

    print('Loss & Accuracy: ', result)


if __name__ == "__main__":
    main()
