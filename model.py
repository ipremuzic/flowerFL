import numpy as np
import tensorflow as tf
from tensorflow import keras

from myutils import get_dataset, get_data_for_client, get_user_at_index 

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

    #model.summary()
    return model


def get_data_for_input(user_index):
    """
    Get data for particualr client, and prepare it for input.
    """
    users, groups, train_data, test_data = get_dataset("femnist")
    #print("train_data_size= {} , test_data_size={}".format(len(train_data), len(test_data)))
    print("User: {}, total users: {}".format(user_index, len(users)))
    user_id = get_user_at_index(user_index, users)
    (x_train, y_train), (x_test, y_test) = get_data_for_client(user_id, users, groups, train_data, test_data)
    
    x_train = np.reshape(x_train, (x_train.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    
    # convert class vectors to binary class matrices 
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)
