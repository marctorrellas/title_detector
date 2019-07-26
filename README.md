Author: Marc Torrellas Socastro\
2019, July

## Overview

The system implemented detects titles in documents, given some features for each title:

    - text
    - is_bold
    - is_italic
    - is_underlined
    - left
    - right
    - top
    - bottom

## Installation


The system required Python3 and conda installed. Step by step recommended installation:

1. Create a new virtual environment using conda
    
    `conda create --name a_name python=3.7`

1. Activate environment
    
    `conda activate a_name` 

1. (From source) This is not really an installation, just allows to create 
the environment (this might take a while)

    `poetry install`
    
1. (From whl, skip if done from source) Install as (this might take a while)

    `pip install title_detector-0.1.0-py3-none-any.whl`
    
1. (Both cases) Install spacy language model

    `python -m spacy download en_core_web_sm`

## Usage

From the root project directory type (this is not required if 
installed with the wheel as package):

    title_detector [command] [possible args]

Help can be retrieved by

    title_detector --help
and

    title_detector [command] --help
   


## Examples

`title_detector train` --> defaults to the sample train data
 
`title_detector train --max_docs 300`

`title_detector detect` --> defaults to the sample test data

`title_detector detect --predicted_data_path sample/train_sections_data_detected.csv`

`title_detector evaluate` --> defaults to the sample test data

`title_detector clean` --> defaults to the default model output location

Note that when using sample data, one needs to be in the root directory 
so that the data can be found

## Testing

TODO