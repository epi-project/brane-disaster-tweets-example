import ast
import os
import shutil
import unittest
from unittest import mock

import pandas as pd

from ..visualization import (keywords_profile, location_profile,
                             plot_bigrams_distribution, prediction_plot,
                             tweets_profile)
from .mock_data import mock_open


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not os.path.exists("location_profile/"):
            os.makedirs("location_profile/")
        if not os.path.exists("keywords_profile/"):
            os.makedirs("keywords_profile/")
        if not os.path.exists("tweets_profile/"):
            os.makedirs("tweets_profile/")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("location_profile/")
        shutil.rmtree("keywords_profile/")
        shutil.rmtree("tweets_profile/")
        os.remove("bigrams_distribution.png")
        os.remove("prediction_plot.png")

    @mock.patch("builtins.open", mock_open)
    def test_location_profile(self):
        data = pd.read_csv("/data/dataset_train.csv")
        result = location_profile(data, store_file=True)
        assert(os.path.exists("location_profile/loc_disaster.png"))
        result = location_profile(data)
        assert(result != "" or result is None)

    @mock.patch("builtins.open", mock_open)
    def test_prediction_plot(self):
        sub_data = pd.read_csv("/data/dataset_submission.csv")
        test_data = pd.read_csv("/data/dataset_test.csv")
        predict_data = pd.merge(sub_data, test_data, on="id")

        result = prediction_plot(predict_data)
        assert(result != "" or result is None)
        result = prediction_plot(predict_data, store_file=True)
        assert(os.path.exists("prediction_plot.png"))

    @mock.patch("builtins.open", mock_open)
    def test_keywords_profile(self):
        train_data = pd.read_csv("/data/dataset_train.csv",
                                 converters={"tokens": ast.literal_eval})
        result = keywords_profile(train_data)
        assert(result != "" or result is None)
        result = keywords_profile(train_data, store_file=True)
        assert(os.path.exists("keywords_profile/disaster_keywords_plot.png"))
        assert(os.path.exists("keywords_profile/disaster_keywords_word_cloud.png"))
        assert(os.path.exists("keywords_profile/keywords_plot.png"))
        assert(os.path.exists("keywords_profile/keywords_word_cloud.png"))
        assert(os.path.exists("keywords_profile/non_disaster_keywords_plot.png"))
        assert(os.path.exists(
            "keywords_profile/non_disaster_keywords_word_cloud.png"))

    @mock.patch("builtins.open", mock_open)
    def test_tweets_profile(self):
        train_data = pd.read_csv("/data/dataset_train.csv",
                                 converters={"tokens": ast.literal_eval})
        result = tweets_profile(train_data)
        assert(result != "" or result is None)
        result = tweets_profile(train_data, store_file=True)
        assert(os.path.exists("tweets_profile/disaster_tweets_plot.png"))
        assert(os.path.exists("tweets_profile/disaster_tweets_word_cloud.png"))
        assert(os.path.exists("tweets_profile/non_disaster_tweets_plot.png"))
        assert(os.path.exists("tweets_profile/non_disaster_tweets_word_cloud.png"))
        assert(os.path.exists("tweets_profile/tweets_plot.png"))
        assert(os.path.exists("tweets_profile/tweets_word_cloud.png"))

    @mock.patch("builtins.open", mock_open)
    def test_plot_bigrams_distribution(self):
        result = plot_bigrams_distribution("dataset_train.csv")
        assert(result != "" or result is None)
        result = plot_bigrams_distribution(
            "dataset_train.csv", store_file=True)
        assert(os.path.exists("bigrams_distribution.png"))


if __name__ == '__main__':
    unittest.main()
