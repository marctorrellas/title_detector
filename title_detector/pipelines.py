# TODO: there's some code duplicated in this file. Refactor
import logging

from title_detector.config import LOGGER_NAME
from title_detector.data_preparation import (
    feature_extraction,
    load_data,
    preprocessing,
)
from title_detector.models.master_model import MasterModel

log = logging.getLogger(LOGGER_NAME)


def run_train_pipeline(data_path, model_path, max_docs, slave_enabled):
    """
    Run a train pipeline
    - load and clean data
    - feature extraction
    - split data
    - train text-based classifier
    - train master classifier using prediction of text-based classifier as a feature
    - save model
    Args:
        data_path:
        model_output:

    Returns:

    """
    # load_data (and check data has no labels)
    df = load_data(data_path, max_docs)

    # pre-process data (remove non-ascii characters, drop 0-length strings)
    # 0-length strings not used for training
    df, _ = preprocessing(df)

    # feature extraction (augment data with text length and basic spaCy features)
    df = feature_extraction(df)

    # df.to_csv("all_data.csv")
    # df = pd.read_csv("all_data.csv", index_col=0)

    # Build master model
    model = MasterModel(slave_enabled)

    model.fit(df)

    # save model
    model.save(model_path)


def run_detect_pipeline(data_path, model_path, predicted_data_path=None):

    # load_data (and check data has no labels)
    df = load_data(data_path, labelled=False)

    # pre-process data (remove non-ascii characters, drop 0-length strings
    # BUT here return indexes to reconstruct at the end)
    df, bypassed_samples = preprocessing(df)

    # feature extraction (augment data with character_length and basic spaCy features)
    df = feature_extraction(df)

    # load model
    model = MasterModel.load(model_path)

    # apply classifier + add bypassed samples
    df = model.predict(df, bypassed_samples)

    # save predictions (into predicted_data_path or to data_path if None is passed)
    log.info("Saving output")
    df.to_csv(predicted_data_path or data_path)
    log.info(f"Output saved to {predicted_data_path or data_path}")


def run_evaluate_pipeline(data_path, model_path, max_docs):

    # load_data (and check data has no labels)
    df = load_data(data_path, max_docs=max_docs)

    # pre-process data (remove non-ascii characters, drop 0-length strings
    # BUT here return samples to reconstruct at the end)
    df, bypassed_samples = preprocessing(df)

    # feature extraction (augment data with character_length and basic spaCy features)
    df = feature_extraction(df)

    # load model
    model = MasterModel.load(model_path)

    # evaluate
    model.evaluate(df, bypassed_samples)
