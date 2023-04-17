#!/usr/bin/python3
'''
Entrypoint for the compute package.
'''
import os
import sys

import json
import yaml

from model import create_submission, train_model
from preprocess import (clean, create_vectors, generate_bigrams,
                        remove_stopwords, tokenize)


def run_dataset_action(cmd: str, filepath: str):
    """
    Runs generic dataset preprocessing action.

    Parameters
    ----------
    cmd: `str`
    The action name.

    filepath: `str`
    The dataset filepath in the DFS.
    """
    return {
        "clean": clean,
        "tokenize": tokenize,
        "remove_stopwords": remove_stopwords,
        "generate_bigrams": generate_bigrams,
    }[cmd](filepath)


def print_output(data: dict):
    """
    Creates a marked section in the standard output
    of the container in order for Brane to isolate the result.

    Parameters
    ----------
    data: `dict`
    Any valid Python dictionary that is YAML serializable.
    """
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")


def main():
    command = sys.argv[1]

    if command == "create_vectors":
        # filepath_train_dataset = os.environ["FILEPATH_TRAIN_DATASET"]
        # filepath_test_dataset = os.environ["FILEPATH_TEST_DATASET"]
        # filepath_train_vectors = os.environ["FILEPATH_TRAIN_VECTORS"]
        # filepath_test_vectors = os.environ["FILEPATH_TEST_VECTORS"]
        # errcode = create_vectors(filepath_train_dataset, filepath_test_dataset,
        #                          filepath_train_vectors, filepath_test_vectors)
        # print_output({"errcode": errcode})

        # Find the input paths to the training & test dataset
        train_dataset = f"{json.loads(os.environ['TRAIN_SET'])}/dataset.csv"
        test_dataset = f"{json.loads(os.environ['TEST_SET'])}/dataset.csv"
        # Generate the path for the vectors
        train_vectors = "/result/train_vectors.pickle"
        test_vectors = "/result/test_vectors.pickle"
        # Call the function
        errcode = create_vectors(train_dataset, test_dataset, train_vectors, test_vectors)
        if errcode != 0: print(f"Uh-oh, 'create_vectors' returned non-zero exit code '{errcode}'", file=sys.stderr); exit(1)

        return

    if command == "train_model":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        filepath_vectors = f"{json.loads(os.environ['FILEPATH_VECTORS'])}/train_vectors.pickle"
        filepath_model = train_model(filepath_dataset, filepath_vectors)
        # print_output({"filepath_model": filepath_model})
        return

    if command == "create_submission":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        filepath_vectors = f"{json.loads(os.environ['FILEPATH_VECTORS'])}/test_vectors.pickle"
        filepath_model = f"{json.loads(os.environ['FILEPATH_MODEL'])}/model.pickle"
        filepath_submission = create_submission(
            filepath_dataset, filepath_vectors, filepath_model)
        # print_output({"filepath_submission": filepath_submission})
        return

    filepath_in = json.loads(os.environ["FILEPATH"])
    filepath_out = run_dataset_action(command, filepath_in)
    # print_output({"filepath": filepath_out})


if __name__ == '__main__':
    main()
