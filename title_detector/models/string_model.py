from pathlib import Path

import numpy as np

from fastai.text import (
    AWD_LSTM,
    TextClasDataBunch,
    TextLMDataBunch,
    language_model_learner,
    load_learner,
    text_classifier_learner,
)
from tqdm import tqdm

from title_detector.config import (
    SLAVE_BATCH_SIZE,
    SLAVE_DROPOUT,
    SLAVE_LR,
    SLAVE_PATH_CLASSIFIER,
    SLAVE_PATH_LM,
)


class StringModel:
    def __init__(self, arch=None, batch_size=None, lr=None, dropout=None, paths=None):

        self.arch = arch or AWD_LSTM
        self.batch_size = batch_size or SLAVE_BATCH_SIZE

        lr = lr or SLAVE_LR
        if not isinstance(lr, list) and isinstance(lr, float):
            lr = [lr, lr]
        else:
            raise ValueError(
                f"Expected lr being float or list of floats, but passed {lr}"
            )
        self.lr_lm, self.lr_class = lr

        dropout = dropout or SLAVE_DROPOUT
        if not isinstance(dropout, list):
            dropout = [dropout, dropout]
        self.dropout_lm, self.dropout_class = dropout

        paths = paths or (SLAVE_PATH_LM, SLAVE_PATH_CLASSIFIER)
        self.path_lm, self.path_class = paths

    def fit(self, df_train, df_val):
        data_lm = self._fit_lm(df_train, df_val)
        return self._fit_class(df_train, df_val, data_lm)

    def _fit_lm(self, df_train, df_val):
        # Language model data
        data_lm = TextLMDataBunch.from_df(train_df=df_train, valid_df=df_val, path="")
        lm_learner = language_model_learner(
            data_lm, self.arch, drop_mult=self.dropout_class
        )
        # train the learner object
        lm_learner.fit_one_cycle(1, self.lr_class)
        # TODO: can we return lm_leaner and load via memory so we don't have to save it?
        lm_learner.save_encoder(self.path_lm.name)
        return data_lm

    def _fit_class(self, df_train, df_val, data_lm):
        n_data = min(len(df_train), len(df_val))
        # Classifier model data
        data_class = TextClasDataBunch.from_df(
            path="",
            train_df=df_train,
            valid_df=df_val,
            vocab=data_lm.train_ds.vocab,
            bs=self.batch_size if self.batch_size < n_data else n_data // 2,
        )
        # train the learner object
        class_learner = text_classifier_learner(
            data_class, self.arch, drop_mult=self.dropout_lm
        )
        class_learner.load_encoder(self.path_lm.name)
        class_learner.fit_one_cycle(1, self.lr_class)
        class_learner.export(self.path_class)

    def predict(self, data):
        path = Path(self.path_class)
        learn_classifier = load_learner(path=path.parent, file=path.name)
        learn_classifier.model.eval()
        preds = []
        # TODO: for sure Fastai has a more efficient way to predict than this
        #  something like:
        #  with concurrent.futures.ProcessPoolExecutor() as executor:
        #    predictions = [i for i in executor.map(predict, x)]
        for text in tqdm(data):
            preds.append(learn_classifier.predict(text)[2].numpy())
        # TODO: return the soft prediction?
        return np.argmax(preds, axis=1)
