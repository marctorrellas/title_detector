import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from title_detector.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_DATA_PATH,
    DEFAULT_TRAIN_DATA_PATH,
    LOG_FILE,
    LOGGER_NAME,
)
from title_detector.utils.logger import awesome_logger
from title_detector.pipelines import run_detect_pipeline, run_evaluate_pipeline, run_train_pipeline
from title_detector.utils.validators import get_validated_max_docs, get_validated_path

log = awesome_logger(LOGGER_NAME, LOG_FILE)


def get_parser():
    # create the top-level parser
    # TODO: try fire! it simplifies this a lot
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(
        help=(
            "Task to be done.\n"
            " - Train: train and save a model\n"
            " - Detect: detect titles on an unlabelled dataset and save to a csv\n"
            " - Evaluate: detect titles on a labelled dataset and score against "
            "labels\n"
            " - Clean: Remove a trained model\n"
        ),
        dest="command",
    )

    parser.add_argument("--data_path", type=str, default=None, help="Data location")

    # create the parser for the "train" command
    parser_train = subparsers.add_parser("train")

    parser_train.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Location where the model is to be saved",
    )

    parser_train.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="Max number of docs to train with. None to use all",
    )

    parser_train.add_argument(
        "--slave_enabled", type=bool, default=False, help="Whether to use a slave model"
    )

    # create the parser for the "detect" command
    parser_detect = subparsers.add_parser("detect")
    parser_detect.add_argument(
        "--data_path", type=str, default=DEFAULT_TEST_DATA_PATH, help="Data location"
    )
    parser_detect.add_argument(
        "--model_path", default=DEFAULT_MODEL_PATH, type=str, help="Model location"
    )
    parser_detect.add_argument(
        "--predicted_data_path",
        default=None,
        type=str,
        help="Path where the data augmented with the detections is to be saved. "
        "Defaults to same as data_path",
    )

    # create the parser for the "evaluate" command
    parser_detect = subparsers.add_parser("evaluate")
    parser_detect.add_argument(
        "--data_path", type=str, default=DEFAULT_TEST_DATA_PATH, help="Data location"
    )
    parser_detect.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model location"
    )
    parser_detect.add_argument(
        "--max_docs",
        type=int,
        default=0,
        help="Max number of docs to train with. None to use all",
    )

    # create the parser for the "clean" command
    parser_clean = subparsers.add_parser("clean")
    parser_clean.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model location"
    )
    return parser


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
    command = args.command
    if command not in ["train", "detect", "evaluate", "clean"]:
        parser.print_help()
        quit()

    if command in ["train", "detect", "evaluate"]:
        default_path = (
            DEFAULT_TRAIN_DATA_PATH if command == "train" else DEFAULT_TEST_DATA_PATH
        )
        data_path = get_validated_path(args.data_path or default_path)
        if command in ["train", "evaluate"]:
            max_docs = get_validated_max_docs(args.max_docs)

    model_path = get_validated_path(args.model_path)

    if command == "train":
        slave_enabled = args.slave_enabled
        run_train_pipeline(data_path, model_path, max_docs, slave_enabled)

    elif command == "detect":
        output_path = args.predicted_data_path
        run_detect_pipeline(data_path, model_path, predicted_data_path=output_path)

    elif command == "evaluate":
        run_evaluate_pipeline(data_path, model_path, max_docs)

    else:  # command == 'clean':
        # TODO: add are u sure? this will remove bla bla bla
        path = get_validated_path(model_path)
        log.info(f"Removing model in {model_path}")
        os.remove(path)
        # TODO: remove also folder if it becomes empty
        log.info("Model removed")


if __name__ == "__main__":
    main(sys.args)
