import os
import pandas as pd
import numpy as np
import pickle
from unittest.mock import patch
from keras.preprocessing.text import Tokenizer
from data.prepdata import main


def test_main():
    # Create a small test csv
    test_csv = "test.csv"
    df = pd.DataFrame(
        {
            "SentimentText": [
                "I love this",
                "I hate this",
                "I feel great",
                "I feel awful",
                "this is bad",
                "awesome",
                "just awful",
            ],
            "Sentiment": [1, 0, 1, 0, 0, 1, 0],
        }
    )
    df.to_csv(test_csv, index=False)

    test_args = ["prepdata.py", test_csv, "--train_size=0.5"]

    with patch("sys.argv", test_args):
        main()

    # Check that output files have been written
    assert os.path.exists("train_valid_test_data.pkl")
    assert os.path.exists("tokenizer.pkl")

    # Load output files
    with open("train_valid_test_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Check that data and tokenizer have the expected contents
    assert isinstance(data, tuple)
    assert len(data) == 6
    assert isinstance(data[0], np.ndarray)
    assert isinstance(data[1], np.ndarray)
    assert isinstance(tokenizer, Tokenizer)

    # Cleanup: delete the test csv file and output files
    os.remove(test_csv)
    os.remove("train_valid_test_data.pkl")
    os.remove("tokenizer.pkl")
