import numpy as np
import os
import pickle
import shutil
from train import lstm_bayes


def test_lstm_bayes(tmpdir):
    # Create a small dataset
    X_train = np.random.randint(0, 999, size=(80, 5))
    y_train = np.array([1 if x[0] % 2 == 0 else 0 for x in X_train])
    X_valid = np.random.randint(0, 999, size=(10, 5))
    y_valid = np.array([1 if x[0] % 2 == 0 else 0 for x in X_valid])
    X_test = np.random.randint(0, 999, size=(10, 5))
    y_test = np.array([1 if x[0] % 2 == 0 else 0 for x in X_test])

    # Save it to a temporary file
    data_file = tmpdir.join("data.pkl")
    with open(data_file, "wb") as f:
        pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test), f)

    # Call the function from lstm_bayes.py
    lstm_bayes.run(
        dataset_location=str(data_file),
        num_words=1000,
        max_trials=2,
        executions_per_trial=1,
        epochs=10,
    )

    # Check that the outputs were created
    assert os.path.isfile("best_hyperparameters.json")
    assert os.path.isfile("sentiment_analysis_model.h5")
    assert os.path.isfile("sentiment_analysis_history.pkl")

    # Remove the output files
    os.remove("best_hyperparameters.json")
    os.remove("sentiment_analysis_model.h5")
    os.remove("sentiment_analysis_history.pkl")
    if os.path.exists("./bayesian_optimization"):
        shutil.rmtree("./bayesian_optimization")
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
