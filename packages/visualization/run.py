#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import ast
import codecs
import os
import sys

import pandas as pd
import json
import yaml

from visualization import (generate_keywords_profile,
                           generate_location_profile, generate_prediction_plot,
                           generate_tweets_profile, keywords_profile,
                           location_profile, plot_bigrams_distribution,
                           prediction_plot, tweets_profile)

dtypes = {
    "id": int,
    "keyword": str,
    "location": str,
    "text": str,
}


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


def visualization_action(
    filepath_train_dataset: str,
    filepath_test_dataset: str,
    filepath_sub_dataset: str,
) -> int:
    """
    Create an Html that contains all the plots based on the test
    and submission datasets.

    Parameters
    ----------
    filepath_test_dataset: `str`
    CSV file containing the test dataset.

    filepath_sub_dataset: `str`
    CSV file containing the submission dataset.

    filepath_bigrams_dataset: `str`
    CSV file containing the bigrams information dataset.
    filepath_train_dataset: `str`
    CSV file containing the training dataset.

    Returns
    -------
    `int` Error code.
    """
    sub_data = pd.read_csv(filepath_test_dataset,
                           converters={"tokens": ast.literal_eval})
    test_data = pd.read_csv(filepath_sub_dataset,
                            converters={"tokens": ast.literal_eval})
    train_data = pd.read_csv(filepath_train_dataset,
                             converters={"tokens": ast.literal_eval})
    predict_data = pd.merge(sub_data, test_data, on="id")

    location_img = location_profile(train_data)
    keywords_imgs = keywords_profile(train_data)
    tweets_imgs = tweets_profile(train_data)
    prdict_img = prediction_plot(predict_data)
    bigrams_img = plot_bigrams_distribution(
        filepath_train_dataset)
    template_html = codecs.open("./result.html", "r", "utf-8")

    result = template_html.read().format(
        prediction_overview=prdict_img,
        keywords_word_cloud=keywords_imgs[1],
        keywords_top30=keywords_imgs[0],
        disaster_keywords_word_cloud=keywords_imgs[3],
        disaster_keywords_top30=keywords_imgs[2],
        non_disaster_keywords_word_cloud=keywords_imgs[5],
        non_disaster_keywords_top30=keywords_imgs[4],
        tweets_text_word_cloud=tweets_imgs[1],
        tweets_text_word_frequency_top30=tweets_imgs[0],
        disaster_tweets_text_word_cloud=tweets_imgs[3],
        disaster_tweets_text_word_frequency_top30=tweets_imgs[2],
        non_disaster_tweets_text_word_cloud=tweets_imgs[5],
        non_disaster_tweets_text_word_frequency_top30=tweets_imgs[4],
        disaster_location_top10=location_img, bigrams_img=bigrams_img)

    try:
        with open("/result/result.html", "w") as f:
            f.write(result)
        return "/result/result.html"
    except IOError as e:
        return ""


def main():
    command = sys.argv[1]

    if command == "visualization_action":
        filepath_train_dataset = f"{json.loads(os.environ['FILEPATH_TRAIN_DATASET'])}/dataset.csv"
        filepath_test_dataset = f"{json.loads(os.environ['FILEPATH_TEST_DATASET'])}/dataset.csv"
        filepath_sub_dataset = f"{json.loads(os.environ['FILEPATH_SUB_DATASET'])}/submission.csv"

        output = visualization_action(
            filepath_train_dataset, filepath_test_dataset,
            filepath_sub_dataset)
        # print_output({"output": output})
        return

    if command == "generate_prediction_plot":
        filepath_test_dataset = f"{json.loads(os.environ['FILEPATH_TEST_DATASET'])}/dataset.csv"
        filepath_sub_dataset = f"{json.loads(os.environ['FILEPATH_SUB_DATASET'])}/submission.csv"
        output = generate_prediction_plot(
            filepath_test_dataset, filepath_sub_dataset)
        # print_output({"output": output})
        return

    if command == "generate_location_profile":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        n_top = json.loads(os.environ["N_TOP"])
        dirs = "/result/location_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_location_profile(filepath_dataset, int(n_top))
        # print_output({"output": output})
        return

    if command == "generate_tweets_profile":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        n_top = json.loads(os.environ["N_TOP"])
        dirs = "/result/tweets_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_tweets_profile(filepath_dataset, int(n_top))
        # print_output({"output": output})
        return

    if command == "generate_keywords_profile":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        n_top = json.loads(os.environ["N_TOP"])
        dirs = "/result/keywords_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_keywords_profile(filepath_dataset, int(n_top))
        # print_output({"output": output})
        return

    if command == "plot_bigrams_distribution":
        filepath_dataset = f"{json.loads(os.environ['FILEPATH_DATASET'])}/dataset.csv"
        n_top_bigrams = json.loads(os.environ["N_TOP_BIGRAMS"])
        filepath_image = plot_bigrams_distribution(
            filepath_dataset, int(n_top_bigrams), True)
        # print_output({"filepath_image": filepath_image})
        return


if __name__ == '__main__':
    main()
