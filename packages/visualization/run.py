#!/usr/bin/python3
'''
Entrypoint for the visualization package.
'''
import os
import sys
import yaml
import codecs
from visualization import *

def visualization_action(input: any):
    predict_data = pd.read_csv(input+'submission.csv')
    test_data = pd.read_csv(input+'test.csv')

    prdict_data = pd.merge(predict_data, test_data, on="id")
    location_img = location_profile(prdict_data)
    keywords_word_cloud_img = keywords_word_cloud(prdict_data)
    keywords_polt_img = keywords_polt(prdict_data)
    tweets_imgs = tweets_profile(prdict_data)
    prdict_img = prediction_plot(predict_data)

    template_html = codecs.open("./result.html","r","utf-8")
    
    result = template_html.read().format(prediction_overview=prdict_img,keywords_word_cloud=keywords_word_cloud_img,keywords_top30=keywords_polt_img,tweets_text_word_cloud=tweets_imgs[1],tweets_text_word_frequency_top30=tweets_imgs[0],disaster_location_top10=location_img)
    
    # We wrap the writing in a try/catch so we may catch any errors
    try:
        # Open the file and write the content
        with open(f"/data/result.html", "w") as f:
            f.write(result)

        return "success"
    # Catch file errors
    except IOError as e:
        # Return the non-zero exit code that they define
        return e.errno


if __name__ == '__main__':
    input = os.environ["INPUT"]
    output = visualization_action(input)
    print(yaml.dump({"output": output}))
