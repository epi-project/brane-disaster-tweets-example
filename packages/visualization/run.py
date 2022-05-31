#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import os
import sys
import yaml
import codecs
from visualization import *

def visualization_action(filepath_test_dataset,filepath_sub_dataset):
    """
    Create an Html that contains all the plots based on the test and submission datasets.

    Parameters
    ----------
    filepath_test_dataset: `str`
    CSV file containing the test dataset.

    filepath_sub_dataset: `str`
    CSV file containing the submission dataset.

    Returns
    -------
    `str` The result of the process.
    """
    
    sub_data = pd.read_csv(filepath_test_dataset)
    test_data = pd.read_csv(filepath_sub_dataset)

    predict_data = pd.merge(sub_data, test_data, on="id")
    location_img = location_profile(predict_data)
    keywords_word_cloud_img = keywords_word_cloud(predict_data)
    keywords_polt_img = keywords_polt(predict_data)
    tweets_imgs = tweets_profile(predict_data)
    prdict_img = prediction_plot(predict_data)

    template_html = codecs.open("./result.html","r","utf-8")
    
    result = template_html.read().format(prediction_overview=prdict_img,keywords_word_cloud=keywords_word_cloud_img,keywords_top30=keywords_polt_img,tweets_text_word_cloud=tweets_imgs[1],tweets_text_word_frequency_top30=tweets_imgs[0],disaster_location_top10=location_img)
    
    try:
        # Open the file and write the content
        with open(f"/data/result.html", "w") as f:
            f.write(result)

        return "success"
    # Catch file errors
    except IOError as e:
        return e.errno


if __name__ == '__main__':
    filepath_test_dataset = os.environ["FILEPATH_TEST_DATASET"]
    filepath_sub_dataset = os.environ["FILEPATH_SUB_DATASET"]
    output = visualization_action(filepath_test_dataset,filepath_sub_dataset)
    print(yaml.dump({"output": output}))
