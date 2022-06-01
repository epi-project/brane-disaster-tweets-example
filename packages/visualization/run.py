#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import codecs
import os
import sys

import pandas as pd
import yaml

from visualization import (keywords_profile, location_profile,
                           plot_bigrams_distribution, prediction_plot,
                           tweets_profile)


def print_output(data: dict):
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")


def visualization_action(
    filepath_test_dataset: str,
    filepath_sub_dataset: str
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

    Returns
    -------
    `int` Error code.
    """
    sub_data = pd.read_csv(filepath_test_dataset)
    test_data = pd.read_csv(filepath_sub_dataset)

    predict_data = pd.merge(sub_data, test_data, on="id")
    location_img = location_profile(predict_data)
    keywords_imgs = keywords_profile(predict_data)
    tweets_imgs = tweets_profile(predict_data)
    prdict_img = prediction_plot(predict_data)

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
        disaster_location_top10=location_img)

    try:
        with open("/data/result.html", "w") as f:
            f.write(result)
        return 0
    except IOError as e:
        return e.errno


def main():
    command = sys.argv[1]

    if command == "visualization_action":
        filepath_test_dataset = "/data/"+os.environ["FILEPATH_TEST_DATASET"]
        filepath_sub_dataset = "/data/"+os.environ["FILEPATH_SUB_DATASET"]
        output = visualization_action(
            filepath_test_dataset, filepath_sub_dataset)
        print_output({"output": output})
        return

    if command == "plot_bigrams_distribution":
        filepath_dataset = os.environ["FILEPATH_DATASET"]
        n_top_bigrams = os.environ["N_TOP_BIGRAMS"]
        filepath_image = plot_bigrams_distribution(
            filepath_dataset, int(n_top_bigrams))
        print_output({"filepath_image": filepath_image})
        return


if __name__ == '__main__':
    main()
