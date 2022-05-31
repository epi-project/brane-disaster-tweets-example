import ast
import unittest
from unittest import mock

from ..actions.preprocess import clean, remove_stopwords, tokenize
from .mock_data import mock_open, mock_to_csv


class TestPreprocessing(unittest.TestCase):

    @mock.patch("builtins.open", mock_open)
    @mock.patch("pandas.DataFrame.to_csv", mock_to_csv)
    def test_clean_urls(self):
        result = clean("dataset_train_raw_http.csv")
        with open(result, "r") as f:
            tweets = list(map(
                lambda x: x.split(',')[-1].strip(), f.readlines()[1:]))
            assert(tweets[0] == "")
            assert(tweets[1] == "")
            assert(tweets[2] == "test")

    @mock.patch("builtins.open", mock_open)
    @mock.patch("pandas.DataFrame.to_csv", mock_to_csv)
    def test_clean_tags(self):
        result = clean("dataset_train_raw_tags.csv")
        with open(result, "r") as f:
            tweets = list(map(
                lambda x: x.split(',')[-1].strip(), f.readlines()[1:]))
            assert(tweets[0] == "")
            assert(tweets[1] == "test")

    @mock.patch("builtins.open", mock_open)
    @mock.patch("pandas.DataFrame.to_csv", mock_to_csv)
    def test_clean_usernames(self):
        result = clean("dataset_train_raw_usernames.csv")
        with open(result, "r") as f:
            tweets = list(map(
                lambda x: x.split(',')[-1].strip(), f.readlines()[1:]))
            assert(tweets[0] == "test")
            assert(tweets[1] == "there")

    @mock.patch("builtins.open", mock_open)
    @mock.patch("pandas.DataFrame.to_csv", mock_to_csv)
    def test_tokenize(self):
        result = tokenize("dataset_train_raw_lemmas.csv")
        with open(result, "r") as f:
            tokens = list(map(
                lambda x: ast.literal_eval(
                    x.split(',', maxsplit=6)[-1].replace("\"", "")),
                f.readlines()[1:]))
            assert(len({'i', 'm', 'up', 'for', 'it'} &
                   set(tokens[0])) == len(set(tokens[0])))
            assert(len({'what', 'are', 'you', 'do'} &
                   set(tokens[1])) == len(set(tokens[1])))

    @mock.patch("builtins.open", mock_open)
    @mock.patch("pandas.DataFrame.to_csv", mock_to_csv)
    def test_remove_stopwords(self):
        tokenized = tokenize("dataset_train_raw_lemmas.csv")
        result = remove_stopwords(tokenized)
        with open(result, "r") as f:
            tokens = list(map(
                lambda x: ast.literal_eval(
                    x.split(',', maxsplit=6)[-1].replace("\"", "")),
                f.readlines()[1:]))
            assert(len(tokens[0]) == 0)
            assert(len(tokens[1]) == 0)
            assert(tokens[2][0] == "disast")
