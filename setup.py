#!/usr/bin/env python

from setuptools import setup, find_packages

from title_detector.config import (
    DEFAULT_TEST_DATA_PATH_FROM_ROOT,
    DEFAULT_TRAIN_DATA_PATH_FROM_ROOT,
)

setup(
    name="title_detector",
    version="1.0",
    description="A detector for titles in OCR outputs",
    author="Marc Torrellas Socastro",
    author_email="marc.torsoc@gmail.com",
    url="https://marctorrellas.github.com",
    packages=find_packages(),
    entry_points={"console_scripts": ["title_detector = title_detector.main:main"]},
    package_data={
        "title_detector": [
            DEFAULT_TRAIN_DATA_PATH_FROM_ROOT,
            DEFAULT_TEST_DATA_PATH_FROM_ROOT,
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "fastai",
        "ipython",
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "spacy",
        "torch",
        "tqdm",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz",
    ],
)