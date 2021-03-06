import argparse
import os

import numpy as np
import tensorflow as tf

from tensorflow import keras

import flwr as fl

from myutils import get_client_at_index, get_data_for_client

from model import create_model, process_data

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class FemnistClient(fl.client.NumPyClient):
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


def start_client(client_id, train_data, test_data):
    # Load and compile Keras model
    model = create_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Get processed client data that can be fed to model
    (x_train, y_train), (x_test, y_test) = process_data(train_data, test_data)
    print("client_id: {} ;(x_train, y_train), (x_test, y_test) = ({}, {}), ({}, {})".format(client_id, len(x_train), len(y_train), len(x_test), len(y_test)))

    # Start Flower client
    client = FemnistClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def main() -> None:
    # Parse command line argument 'cid'
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=int, choices=range(0, 100), required=True)
    args = parser.parse_args()

    # Get dataset and start client
    c_train_data, c_test_data = get_data_for_client(args.cid)
    start_client(args.cid, c_train_data, c_test_data)


if __name__ == "__main__":
    main()
