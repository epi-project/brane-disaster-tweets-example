#!/usr/bin/python3
'''
Entrypoint for the compute package.
'''
import os
import sys

import yaml

from actions.preprocess import clean, remove_stopwords, tokenize


def run_action(cmd: str, filepath: str):
    return {
        "clean": clean,
        "tokenize": tokenize,
        "remove_stopwords": remove_stopwords
    }[cmd](filepath)


def main():
    command = sys.argv[1]
    filepath_in = os.environ["FILEPATH"]
    filepath_out = run_action(command, filepath_in)
    print("--> START CAPTURE")
    print(yaml.dump({"filepath": filepath_out}))
    print("--> END CAPTURE")


if __name__ == '__main__':
    main()
