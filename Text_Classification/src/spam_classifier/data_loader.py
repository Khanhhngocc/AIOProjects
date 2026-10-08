import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from spam_classifier.config import DATASET_PATH, MESSAGE_COLUMN, LABEL_COLUMN, TEST_SIZE, RANDOM_STATE


def load_dataset(dataset_path=DATASET_PATH):
    """Load the spam classification dataset."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please place 2cls_spam_text_cls.csv inside the data folder."
        )

    df = pd.read_csv(dataset_path)

    required_columns = {MESSAGE_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    messages = df[MESSAGE_COLUMN].astype(str).tolist()
    labels = df[LABEL_COLUMN].astype(str).tolist()

    return df, messages, labels


def create_metadata(messages, labels):
    """Create metadata for each message."""

    return [
        {
            "index": i,
            "message": message,
            "label": label
        }
        for i, (message, label) in enumerate(zip(messages, labels))
    ]


def split_dataset(messages, labels, embeddings):
    """Split embeddings and metadata into train and test sets."""

    metadata = create_metadata(messages, labels)

    train_indices, test_indices = train_test_split(
        np.arange(len(messages)),
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    X_train_emb = embeddings[train_indices]
    X_test_emb = embeddings[test_indices]

    train_metadata = [metadata[i] for i in train_indices]
    test_metadata = [metadata[i] for i in test_indices]

    return X_train_emb, X_test_emb, train_metadata, test_metadata