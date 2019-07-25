import re
import logging

import numpy as np
import pandas as pd
import spacy
from torch.utils.data import Dataset
from tqdm import tqdm

from title_detector.config import COLUMNS, SPACY_FEATURES, LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


def load_data(data_path, max_docs=None, labelled=True):
    log.info("Loading and enriching data")
    df = pd.read_csv(data_path, encoding="latin1", nrows=max_docs)
    df.columns = [to_snake(col) for col in df.columns]
    # TODO: maybe fail if labelled=False and there are labels?
    # TODO: maybe the constant columns should be detected automatically?
    filtered_columns = COLUMNS + (["label"] if labelled else [])
    check_columns(df, filtered_columns)
    df = df[filtered_columns]
    return df


def to_snake(s):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def check_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Expect column {col} not present in data")


def preprocessing(df):
    # remove non-ascii
    df["text"] = df["text"].str.replace(r"[^\x00-\x7F]+", " ")
    #  filter columns with zero-length, and save them for later
    zero_length_samples = df.loc[df["text"] == " "]
    df.drop(zero_length_samples.index, inplace=True)
    return df, zero_length_samples


def feature_extraction(df):
    """
    Adds features to the dataframe `df`
    """
    # Add character length
    df["char_length"] = df["text"].str.len()

    # Load spaCy model
    # TODO: load lg? if so, change requirements.txt
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

    # Tokenise documents
    documents = []
    for doc in tqdm(df["text"]):
        documents.append(nlp(doc))
    df["token_length"] = list(map(len, documents))

    # Extract spaCy-based features
    # TODO: maybe doing this on the dataframe directly is more efficient? there are two loops here
    features = np.zeros((len(documents), len(SPACY_FEATURES)))
    for idx, doc in enumerate(tqdm(documents)):
        features[idx, :] = doc.to_array(SPACY_FEATURES).sum(axis=0) / len(doc)
    for idx, feature in enumerate(SPACY_FEATURES):
        df[feature.lower()] = features[:, idx]
    return df


class LabelledDataset(Dataset):
    """Dataset child with features and target"""

    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.target[index]
