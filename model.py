import numpy as np
import tensorflow as tf
from tensorflow import keras

# Model / data parameters
num_classes = 62
input_shape = (28, 28, 1)

# Define model
def create_model():
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Uncomment to print model summary
    #model.summary()

    return model


def process_data(train_data, test_data):
    """
    Process client's dataset so it is ready for input.
    """
    raw_train_x = train_data['x']
    raw_train_y = train_data['y']
    raw_test_x = test_data['x']
    raw_test_y = test_data['y']

    x_train = np.array(raw_train_x)
    y_train = np.array(raw_train_y)
    x_test = np.array(raw_test_x)
    y_test = np.array(raw_test_y)

    # reshape data to (num_samples, input_shape)
    x_train = np.reshape(x_train, (x_train.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], input_shape[0], input_shape[1], input_shape[2]))

    # convert class vectors to binary class matrices 
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)