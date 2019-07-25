import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score

from title_detector.config import LOGGER_NAME

log = logging.getLogger(LOGGER_NAME)


def get_validated_max_docs(max_docs):
    try:
        max_docs = int(max_docs)
    except ValueError:
        raise ValueError("Invalid max docs argument")
    if max_docs < 0:
        raise ValueError(
            f"max_docs must be a positive integer , but {max_docs} was passed"
        )

    return max_docs if max_docs else None


def get_validated_path(path):
    path = Path(path)
    if not path.exists:
        raise FileNotFoundError(f"Path {path} not found")
    return path


def get_validated_scorer(scorer):
    if scorer is None:
        return roc_auc_score
    if not callable(scorer):
        raise ValueError(f"Scorer must be a callable, but {scorer} was passed")
    return scorer


def get_validated_verbose(verbose):
    if not isinstance(verbose, int) or verbose < 0:
        raise ValueError(f"Verbose must be a non-neg integer, but {verbose} was passed")
    return verbose
