import argparse
import os

import numpy as np
import tensorflow as tf

from tensorflow import keras

import flwr as fl

from myutils import * 

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Model / data parameters
num_classes = 62
input_shape = (28, 28, 1)

#define model
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

    model.summary()
    return model


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train, self.y_train, batch_size, epochs, validation_split=0.1
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(input_shape=(32, 32, 3), weights=None, classes=10)
    model = create_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    #(x_train, y_train), (x_test, y_test) = load_partition(args.partition)
    users, groups, train_data, test_data = get_dataset("femnist")
    print("Partition: {} of Users: {}".format(args.partition, len(users)))
    #print("train_data_size= {} , test_data_size={}".format(len(train_data), len(test_data)))
    user_id = get_user_at_index(args.partition, users)
    (x_train, y_train), (x_test, y_test) = get_data_for_client(user_id, users, groups, train_data, test_data)
    
    # Scale images to the [0, 1] range
    #x_train = x_train.astype("float32") / 255
    #x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #print("x_train shape:", x_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
    #print("x_train shape after reshape:", x_train.shape)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #print("x_train shape after expand:", x_train.shape)
    #print(x_train.shape[0], "train samples")
    #print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices 
    #print("y_train shape:", y_train.shape)
    #print("y_test shape:", y_test.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #print("y_train shape after reshape:", y_train.shape)
    #print("y_test shape after reshape:", y_test.shape)
    
    print("(x_train, y_train), (x_test, y_test) = ({}, {}), ({}, {})".format(len(x_train), len(y_train), len(x_test), len(y_test)))

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()
