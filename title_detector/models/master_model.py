import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from title_detector.config import COLUMNS, LOGGER_NAME
from title_detector.data_preparation import LabelledDataset
from title_detector.models.pytorch_model import PytorchModel
from title_detector.models.string_model import StringModel

log = logging.getLogger(LOGGER_NAME)


class MasterModel:
    def __init__(self, slave_enabled):
        self.slave_enabled = slave_enabled
        if slave_enabled:
            self.slave_model = StringModel()
        self.pytorch_model = PytorchModel(slave_enabled)

    def fit(self, data):
        # split train1, val1 (aka train2), val2 (see documentation)
        x_train1, x_val1, x_val2, y_train1, y_val1, y_val2 = self._split_data(data)

        if self.slave_enabled:
            # Train slave model
            self._fit_slave(x_train1, x_val1, y_train1, y_val1)
            # Get data to train master model: train with val1, validating with val2
            x_val1 = x_val1.assign(prediction_slave=self._predict_slave(x_val1))
            x_val2 = x_val2.assign(prediction_slave=self._predict_slave(x_val2))

        # train master model
        self._fit_master(
            x_val1.drop("text", axis=1), x_val2.drop("text", axis=1), y_val1, y_val2
        )

    def _split_data(self, df):
        log.info("Splitting data")
        x, y = df.drop("label", axis=1), df["label"]
        split_kwargs = {"train_size": 0.8, "test_size": 0.2, "shuffle": True}
        x_train1, x_val2, y_train1, y_val2 = train_test_split(
            x, y, stratify=y, random_state=8, **split_kwargs
        )
        split_kwargs = {"train_size": 0.6, "test_size": 0.4, "shuffle": True}
        x_train1, x_val1, y_train1, y_val1 = train_test_split(
            x_train1, y_train1, stratify=y_train1, random_state=8, **split_kwargs
        )
        return x_train1, x_val1, x_val2, y_train1, y_val1, y_val2

    def _fit_slave(self, x_train1, x_val1, y_train1, y_val1):
        log.info("Training slave model")
        self.slave_model.fit(
            pd.concat((y_train1, x_train1["text"]), axis=1),
            pd.concat((y_val1, x_val1["text"]), axis=1),
        )

    def _fit_master(self, x_train, x_val, y_train, y_val):
        log.info("Training master model")
        train_data = self._get_labelled_dataset(x_train, y_train)
        val_data = self._get_labelled_dataset(x_val, y_val)
        self.pytorch_model.fit_model(train_data, val_data)

    def _get_labelled_dataset(self, x_train, y_train):
        return LabelledDataset(
            torch.tensor(x_train.values.astype(np.float)),
            torch.tensor(y_train.values.astype(np.int)),
        )

    def save(self, path):
        log.info(f"Saving model to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        log.info("Loading model")
        return joblib.load(path)

    def predict(self, data, bypassed_samples, hard=True):
        """High level predict using both slave and master"""
        log.info("Predicting")
        if self.slave_enabled:
            data = data.assign(predicted_slave=self._predict_slave(data))
        features = torch.tensor(data.drop("text", axis=1).values.astype(np.float))
        predictions = self.pytorch_model.predict(features, hard=hard)
        return self._build_output_df(data, bypassed_samples, predictions)

    def _predict_slave(self, data):
        return self.slave_model.predict(data["text"])

    def _build_output_df(self, data, bypassed_samples, predictions):
        """Insert bypassed samples with is_title=0 into data in the same order as
        originally, and is_title equal to the prediction in `predictions`. Note that
        the output df here is filtering some input columns (FontType and Unnamed*)"""
        data = data.assign(detected_as_title=predictions)
        bypassed_samples = bypassed_samples.assign(detected_as_title=False)
        data = pd.concat((data, bypassed_samples), axis=0, sort=False).sort_index()
        return data[COLUMNS + ["detected_as_title"]]

    def evaluate(self, df, bypassed_samples):
        """ """
        log.info("Evaluating model")
        x, y = df.drop("label", axis=1), df["label"]
        if self.slave_enabled:
            x = x.assign(prediction_slave=self._predict_slave(x))
        data = self._get_labelled_dataset(x.drop("text", axis=1), y)
        self.pytorch_model.eval_model_for_set(
            data, title="", bypassed=len(bypassed_samples), print_cm=True
        )
