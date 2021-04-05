from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
from tensorflow import keras

from model import create_model, get_data_for_input


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = create_model()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=5,
        min_eval_clients=3,
        min_available_clients=5,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 2}, strategy=strategy)


# TODO: fix
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), _ = get_data_for_input(0)


    # Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_train, y_train)
        return loss, accuracy

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform one local evaluation steps on each client (i.e., use one
    batches) during rounds one to three, then increase to two local
    evaluation steps.
    """
    val_steps = 1 if rnd < 4 else 2
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
