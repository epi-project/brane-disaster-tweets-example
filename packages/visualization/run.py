#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import codecs
import os
import sys
import ast
import pandas as pd
import yaml

from visualization import (keywords_profile, location_profile,plot_bigrams_distribution, prediction_plot,tweets_profile,generate_location_profile,generate_prediction_plot,generate_tweets_profile,generate_keywords_profile)

dtypes = {
    "id": int,
    "keyword": str,
    "location": str,
    "text": str,
}
def print_output(data: dict):
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")

def visualization_action(
    filepath_test_dataset: str,
    filepath_train_dataset: str,
    filepath_sub_dataset: str,
    filepath_bigrams_dataset: str,
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
            filepath_bigrams_dataset)
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
        disaster_location_top10=location_img,bigrams_img=bigrams_img)

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
        filepath_train_dataset = "/data/"+os.environ["FILEPATH_TRAIN_DATASET"]
        filepath_bigrams_dataset = os.environ["FILEPATH_BIGRAMS_DATASET"]
        filepath_sub_dataset = "/data/"+os.environ["FILEPATH_SUB_DATASET"]

        # filepath_test_dataset = "test_clean_tokenized_nostopwords.csv"
        # filepath_train_dataset = "train_clean_tokenized_nostopwords.csv"
        # filepath_dataset = "train_clean_tokenized_nostopwords.csv"
        # filepath_sub_dataset = "submission.csv"
        # filepath_bigrams_dataset = "train_clean_bigrams.csv"
        # n_top_bigrams = 15
        # n_top=10
        output = visualization_action(
            filepath_test_dataset,filepath_train_dataset, filepath_sub_dataset,filepath_bigrams_dataset)
        print_output({"output": output})
        return

    if command == "generate_prediction_plot":
        filepath_test_dataset = "/data/"+os.environ["FILEPATH_TEST_DATASET"]
        filepath_sub_dataset = "/data/"+os.environ["FILEPATH_SUB_DATASET"]
        output = generate_prediction_plot(
            filepath_test_dataset, filepath_sub_dataset)
        print_output({"output": output})
        return

    if command == "generate_location_profile":
        filepath_dataset = "/data/"+os.environ["FILEPATH_DATASET"]
        n_top = os.environ["N_TOP"]
        dirs = "/data/location_profile"
        # dirs = "/Users/user/Documents/wscbs/disaster-tweets-brane/packages/visualization/data/location_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_location_profile(filepath_dataset,n_top)
        print_output({"output": output})
        return
    
    if command == "generate_tweets_profile":
        filepath_dataset = "/data/"+os.environ["FILEPATH_DATASET"]
        n_top = os.environ["N_TOP"]
        dirs = "/data/tweets_profile"
        # dirs = "/Users/user/Documents/wscbs/disaster-tweets-brane/packages/visualization/data/tweets_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_tweets_profile(filepath_dataset,n_top)
        print_output({"output": output})
        return
    
    if command == "generate_keywords_profile":
        filepath_dataset = "/data/"+os.environ["FILEPATH_DATASET"]
        n_top = os.environ["N_TOP"]
        dirs = "/data/keywords_profile"
        # dirs = "/Users/user/Documents/wscbs/disaster-tweets-brane/packages/visualization/data/keywords_profile"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        output = generate_keywords_profile(filepath_dataset,n_top)
        print_output({"output": output})
        return
    
    if command == "plot_bigrams_distribution":
        filepath_dataset = os.environ["FILEPATH_DATASET"]
        n_top_bigrams = os.environ["N_TOP_BIGRAMS"]
        filepath_image = plot_bigrams_distribution(
            filepath_dataset, int(n_top_bigrams),True)
        print_output({"filepath_image": filepath_image})
        return


if __name__ == '__main__':
    main()
