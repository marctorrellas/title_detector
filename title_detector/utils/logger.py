import logging
import time
from pathlib import Path


def awesome_logger(logger_name="AWESOME_LOGGER", logfname="awesome_log.txt"):
    logger = logging.getLogger(logger_name)
    logging.Formatter.converter = time.gmtime
    Path(logfname).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(logfname), logging.StreamHandler()]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger
