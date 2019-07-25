#  TODO: turn this into a raw txt file with proper config handlers?
from pathlib import Path


# Logger
LOGGER_NAME = "TITLE_LOGGER"
LOG_FILE = Path(__file__).parent / "log/title_detector.log"
VERBOSE = 10  # frequency (in epochs) for showing accuracy and loss during training
SHOW_CONFUSION_MATRICES = False

# General
DEFAULT_TRAIN_DATA_PATH_FROM_ROOT = "sample/train_sections_data.csv"
DEFAULT_TEST_DATA_PATH_FROM_ROOT = "sample/test_sections_data.csv"
DEFAULT_TRAIN_DATA_PATH = Path(__file__).parent / DEFAULT_TRAIN_DATA_PATH_FROM_ROOT
DEFAULT_TEST_DATA_PATH = Path(__file__).parent / DEFAULT_TEST_DATA_PATH_FROM_ROOT
DEFAULT_MODEL_PATH = "models/title_detector.joblib"
COLUMNS = [
    "text",
    "is_bold",
    "is_italic",
    "is_underlined",
    "left",
    "right",
    "top",
    "bottom",
]
SPACY_FEATURES = ["IS_STOP", "IS_UPPER", "IS_LOWER", "IS_DIGIT", "IS_PUNCT", "IS_ASCII"]

SCORER = None  # needs to be a callable

# Slave / String layer params
SLAVE_BATCH_SIZE = 128
SLAVE_PATH_LM = Path("models/lm")
SLAVE_PATH_CLASSIFIER = Path("models/text_classifier")
SLAVE_LR = 1e-2  # this can be a list or a float
SLAVE_DROPOUT = 0.7  # this can be a list or a float


# Master / Dense / Pytorch Layer params
MASTER_EPOCHS = 50
MASTER_BATCH_SIZE = 128
MASTER_LR = 1e-2
# original columns - text + spaCy features + token&char length
# optionally we will sum 1 to this if text-based prediction used (slave_enabled=True)
MASTER_INPUT_NUM_UNITS = len(COLUMNS) - 1 + len(SPACY_FEATURES) + 2
MASTER_HIDDEN_NUM_UNITS = MASTER_INPUT_NUM_UNITS // 2
