#!/usr/bin/python3
'''
Entrypoint for the compute package.
'''
import os
import sys

import yaml

from actions.model import create_submission, train_model
from actions.preprocess import (clean, create_vectors, remove_stopwords,
                                tokenize)


def run_dataset_action(cmd: str, filepath: str):
    return {
        "clean": clean,
        "tokenize": tokenize,
        "remove_stopwords": remove_stopwords,
    }[cmd](filepath)


def print_output(data: dict):
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")


def main():
    command = sys.argv[1]

    if command == "create_vectors":
        filepath_train_dataset = os.environ["FILEPATH_TRAIN_DATASET"]
        filepath_test_dataset = os.environ["FILEPATH_TEST_DATASET"]
        filepath_train_vectors = os.environ["FILEPATH_TRAIN_VECTORS"]
        filepath_test_vectors = os.environ["FILEPATH_TEST_VECTORS"]
        errcode = create_vectors(filepath_train_dataset, filepath_test_dataset,
                                 filepath_train_vectors, filepath_test_vectors)
        print_output({"errcode": errcode})
        return

    if command == "train_model":
        filepath_dataset = os.environ["FILEPATH_DATASET"]
        filepath_vectors = os.environ["FILEPATH_VECTORS"]
        filepath_model = train_model(filepath_dataset, filepath_vectors)
        print_output({"filepath_model": filepath_model})
        return

    if command == "create_submission":
        filepath_dataset = os.environ["FILEPATH_DATASET"]
        filepath_vectors = os.environ["FILEPATH_VECTORS"]
        filepath_model = os.environ["FILEPATH_MODEL"]
        filepath_submission = create_submission(
            filepath_dataset, filepath_vectors, filepath_model)
        print_output({"filepath_submission": filepath_submission})
        return

    filepath_in = os.environ["FILEPATH"]
    filepath_out = run_dataset_action(command, filepath_in)
    print_output({"filepath": filepath_out})


if __name__ == '__main__':
    main()
