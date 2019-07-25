import logging

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.nn import Linear
from torch.utils.data import DataLoader

from title_detector.config import (
    LOGGER_NAME,
    MASTER_BATCH_SIZE,
    MASTER_EPOCHS,
    MASTER_HIDDEN_NUM_UNITS,
    MASTER_INPUT_NUM_UNITS,
    MASTER_LR,
    SCORER,
    SHOW_CONFUSION_MATRICES,
    VERBOSE,
)
from title_detector.utils.analysis import my_confusion_matrix
from title_detector.utils.validators import get_validated_scorer, get_validated_verbose

log = logging.getLogger(LOGGER_NAME)


class PytorchModel:
    """
    TODO: think a better name for this (wrapper?)
    TODO: docstring
    """

    def __init__(self, slave_enabled):
        self.epochs = MASTER_EPOCHS
        self.batch_size = MASTER_BATCH_SIZE
        self.lr = MASTER_LR
        self.verbose = get_validated_verbose(VERBOSE)
        self.scorer = get_validated_scorer(SCORER)
        self.model = PytorchModule(slave_enabled)
        self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit_model(self, train_data, val_data, return_history=False):
        train_hist, val_hist = [], []
        # TODO: add patience
        for epoch in range(self.epochs):

            # TODO: review how to set the seed for this for reproducibility
            train_loader = DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True
            )
            self._fit_epoch(train_loader)
            self.model.eval()
            if self.verbose and (epoch + 1) % self.verbose == 0:
                log.info(f"epoch {epoch + 1}/{self.epochs}")
                loss, score = self.eval_model_for_set(train_data, "Train")
                train_hist.append((loss, score))
                loss, score = self.eval_model_for_set(val_data, "Val")
                val_hist.append((loss, score))

        # TODO: this is just a placeholder, not used at the moment
        if return_history:
            return train_hist, val_hist

    def _fit_epoch(self, train_loader):
        self.model.train()
        for batch in train_loader:
            batch_features, batch_tags = batch

            # Clear gradients before each instance
            # (safer using model than optimizer here)
            self.model.zero_grad()
            # Run our forward pass: predictions and loss
            logits = self.model(batch_features)
            loss = self.model.loss(logits, batch_tags)
            # Compute the gradients
            loss.backward()
            # Update the parameters
            self.optimizer.step()

    def eval_model_for_set(self, data, title, bypassed=0, print_cm=False):
        """

        Args:
            data (LabelledDataset):
            title (str): name of the set
            bypassed (float): number of samples classified as negative using domain
                knowledge
            print_cm (bool): whether to print the confusion matrices

        Returns:

        """
        self.model.eval()
        logits = self.model(data.features)
        # TODO: add bypassed to logits here, so the loss is computed with them too
        loss = self.model.loss(logits, data.target).item()
        # TODO: save results of analysis, e.g. a png for the roc_curve, for posterior analysis
        # fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
        predictions = self.model.predict_proba(logits)
        predictions = np.concatenate((predictions, np.zeros(bypassed)))
        target = np.concatenate((data.target, np.zeros(bypassed)))
        score = self.scorer(target, predictions)
        conf_matrix = my_confusion_matrix(target, (predictions > 0.5).astype(int))
        log.info(f"{title} (Loss, Score): {loss:.4}, {score:.4}")
        if SHOW_CONFUSION_MATRICES or print_cm:
            print(conf_matrix)
        return loss, score

    def predict(self, features, hard):
        self.model.eval()
        logits = self.model(features)
        predictions = self.model.predict_proba(logits)
        if hard:
            predictions = (predictions > 0.5).astype(bool)
        return predictions


class PytorchModule(torch.nn.Module):
    """
        TODO: think of a better name for this
        TODO: docstring
    """

    def __init__(self, slave_enabled):
        super(PytorchModule, self).__init__()
        self.linear_in = Linear(
            MASTER_INPUT_NUM_UNITS + bool(slave_enabled), MASTER_HIDDEN_NUM_UNITS
        )
        self.linear_out = Linear(MASTER_HIDDEN_NUM_UNITS, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, features):
        hidden = torch_f.relu(self.linear_in(features), inplace=True)
        return self.linear_out(hidden)

    def loss(self, predictions, target):
        return self.loss_fn(predictions, target)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch_f.softmax(x, dim=1).numpy()[:, 1]
