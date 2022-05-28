#!/usr/bin/python3
'''
Entrypoint for the compute package.
'''
import os
import sys

import yaml

from actions.preprocess import clean, remove_stopwords, tokenize


def run_action(cmd: str, input: any):
    return {
        "clean": clean,
        "tokenize": tokenize,
        "remove_stopwords": remove_stopwords
    }[cmd](input)


def main():
    command = sys.argv[1]
    argument = os.environ["INPUT"]
    output = run_action(command, argument)
    print(yaml.dump({"output": output}))


if __name__ == '__main__':
    main()
