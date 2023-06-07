import unicodedata
import re
import argparse

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def unicode_to_ascii(s):
    """Normalize and convert unicode characters to ASCII."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    """Preprocess a sentence to prepare for tokenization."""
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?!,¿]+", " ", w)
    w = w.strip()
    w = "<start> " + w + " <end>"
    return w


def read_and_preprocess_data(file_path):
    """Load data from csv file and preprocess the text."""
    df = pd.read_csv(file_path, quotechar='"', error_bad_lines=False)
    df["ProcessedText"] = df["SentimentText"].apply(preprocess_sentence)
    return df["ProcessedText"], df["Sentiment"]


def tokenize_and_pad_texts(
    X_train, X_valid, X_test, num_words=1000, max_sequence_length=100
):
    """Tokenize and pad the text data."""
    tokenizer = Tokenizer(
        num_words=num_words, lower=False, filters="", oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(X_train)

    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_valid = tokenizer.texts_to_sequences(X_valid)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(
        sequences_train, maxlen=max_sequence_length, padding="post"
    )
    X_valid_padded = pad_sequences(
        sequences_valid, maxlen=max_sequence_length, padding="post"
    )
    X_test_padded = pad_sequences(
        sequences_test, maxlen=max_sequence_length, padding="post"
    )

    return X_train_padded, X_valid_padded, X_test_padded, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script preprocesses sentiment text data and prepares it \
        for training an LSTM model. It assumes the data is given in csv format \
        with a 'SentimentText' column for the text and 'Sentiment' column for \
        labels (positive=1, negative=0). Either prepare the data in this format \
        or adjust the script."
    )
    parser.add_argument(
        "file_path",
        help="Path to the csv file containing sentiment data.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Train size to data size ratio in range (0, 1). Default is 0.8.",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.5,
        help=(
            "Validation size to non-train data size ratio in range (0, 1). \
            Default is 0.5."
        ),
    )
    parser.add_argument(
        "--num_words",
        type=int,
        default=1000,
        help=(
            "Maximum number of words to keep, based on word frequency.\
            Default is 1000."
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=100,
        help=(
            "Maximum length for all sequences. If a sequence is shorter than \
            max_sequence_length, it will be padded. Default is 100."
        ),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    features, labels = read_and_preprocess_data(args.file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=args.train_size, random_state=42, stratify=labels
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test,
        y_test,
        train_size=args.validation_size,
        random_state=42,
        stratify=y_test,
    )

    X_train_padded, X_valid_padded, X_test_padded, tokenizer = tokenize_and_pad_texts(
        X_train.tolist(), X_valid.tolist(), X_test.tolist()
    )

    # Save necessary data for training and inference
    with open("train_valid_test_data.pkl", "wb") as f:
        pickle.dump(
            (
                X_train_padded,
                np.array(y_train.tolist()),
                X_valid_padded,
                np.array(y_valid.tolist()),
                X_test_padded,
                np.array(y_test.tolist()),
            ),
            f,
        )
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


if __name__ == "__main__":
    main()
