import argparse
import time
import json

import pickle

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
import keras_tuner
from keras_tuner import BayesianOptimization
from tensorflow.keras.callbacks import TensorBoard


class HyperModel(keras_tuner.HyperModel):
    def __init__(self, num_words, max_sequence_length):
        super().__init__()
        self.num_words = num_words
        self.max_sequence_length = max_sequence_length

    def build(self, hp):
        model = Sequential()
        model.add(
            Embedding(
                input_dim=self.num_words,
                output_dim=hp.Int("output_dim", min_value=32, max_value=512, step=32),
                input_length=self.max_sequence_length,
            )
        )
        model.add(LSTM(hp.Int("lstm_units", min_value=32, max_value=128, step=32)))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = hp.Choice("optimizer", values=["adam", "rmsprop"])
        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-5,
            max_value=1e-2,
            sampling="LOG",
            default=1e-3,
        )

        if optimizer == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        batch_size = hp.Int("batch_size", min_value=32, max_value=512, step=32)
        self.batch_size = batch_size

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.get("batch_size"),
            **kwargs,
        )


def run(
    dataset_location,
    num_words,
    max_trials=10,
    executions_per_trial=1,
    epochs=200,
):
    # Load the data
    with open(dataset_location, "rb") as f:
        X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)

    max_sequence_length = X_train.shape[1]

    hypermodel = HyperModel(num_words, max_sequence_length)

    # Initialize the tuner
    tuner = BayesianOptimization(
        hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        seed=42,
        directory="bayesian_optimization",
        project_name="sentiment_analysis",
    )

    # Perform the hyperparameter search
    tuner.search(
        X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), verbose=1
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(
        f"""
    The hyperparameter search is complete.
    The optimal number of units in the LSTM layer is {best_hps.get('lstm_units')},
    the best optimizer is {best_hps.get('optimizer')}
    and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    """
    )

    # Save the optimal hyperparameters to a json file
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_hps.values, f)

    # Retrain the model with the optimal hyperparameters and the whole dataset
    model = tuner.hypermodel.build(best_hps)
    log_dir = "./logs/sentiment/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        np.concatenate((X_train, X_valid)),
        np.concatenate((y_train, y_valid)),
        epochs=epochs,
        callbacks=[tensorboard_callback],
    )

    # Save the model and the history
    model.save("sentiment_analysis_model.h5")
    with open("sentiment_analysis_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"The crossentropy on test data is {loss} and the accuracy is {accuracy}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for sentiment analysis training with Bayesian \
        hyperparameter tuning"
    )
    parser.add_argument(
        "dataset_location",
        type=str,
        help="Location of the pickled post-processed dataset",
    )
    parser.add_argument("num_words", type=int, help="Number of words in the dictionary")
    parser.add_argument(
        "--max_trials",
        type=int,
        default=10,
        help="Max number of hyperparameter trials. Default is 10.",
    )
    parser.add_argument(
        "--executions_per_trial",
        type=int,
        default=1,
        help="Number of executions per trial. Default is 1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of epochs to train. Default is 200.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        args.dataset_location,
        args.num_words,
        args.max_trials,
        args.executions_per_trial,
        args.epochs,
    )


if __name__ == "__main__":
    main()
