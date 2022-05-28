#!/usr/bin/python3

get_ipython().system('pip install pandas numpy seaborn matplotlib missingno wordcloud nltk pandas-profiling[notebook] lxml')
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import base64
# from pandas_profiling import ProfileReport
import re
import io
import codecs

def keywords_polt(data):
    fig = plt.figure(figsize=(25,12))
    data["keyword"].value_counts(sort=True, dropna=True).nlargest(30).plot.bar()
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)

def keywords_word_cloud(data):
    fig = plt.figure(figsize = (15, 15), facecolor = None)
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(data["keyword"].value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image                      
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return  ' '.join(words)

def clean_text(text):
    clean_data = text.lower()
    clean_data = re.sub(r"http.*", " ",clean_data)
    clean_data = re.sub(r"[0-9]+", " ",clean_data)
    clean_data = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ",clean_data)
    return clean_data

def tweets_wordcloud(data):
    fig = plt.figure(figsize = (20, 20), facecolor = None)
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(pd.Series(' '.join(data["token_data"]).split()).value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image                      
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)

def tweets_word_plot(data):
    fig = plt.figure(figsize=(25,12))
    pd.Series(' '.join(data["token_data"]).split()).value_counts(sort=True).nlargest(30).plot.bar(rot=0)
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)

    
def location_plot(data, n):
    fig = plt.figure(figsize=(25,12))
    data.nlargest(n).plot.bar(rot=0)
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)

def prediction_plot(data):
    fig = plt.figure(figsize=(25,12))
    data["target"].value_counts(sort=True, dropna=True).plot.pie(autopct="%1.1f%%",rot=0 , fontsize=30)

    return fig_to_base64(fig)

def tweets_profile(data):
    data["clean_data"] = data["text"].apply(clean_text)
    tokenizer = RegexpTokenizer(r'\w+')
    data["token_data"] = data["clean_data"].apply(tokenizer.tokenize)
    data["token_data"] = data["token_data"].apply(remove_stopwords)
    image_tweets_polt = tweets_word_plot(data)
    image_tweets_word_cloud = tweets_wordcloud(data)
    return [image_tweets_polt,image_tweets_word_cloud]
    
    
def location_profile(data):
    location_map = {"Worldwide":"Earth","Everywhere":"Earth","United Kingdom":"UK","Manchester":"UK","London, England":"UK","Memphis, TN":"USA","Dallas, TX":"USA","Oklahoma City, OK":"USA","San Diego, CA":"USA","Pennsylvania, USA":"USA","Texas":"USA","Seattle":"USA","Chicago":"USA","Florida":"USA","NYC":"USA","San Francisco":"USA","Atlanta, GA":"USA","San Francisco, CA":"USA","New York City":"USA","US":"USA","Denver, Colorado":"USA","Chicago, IL":"USA","Nashville, TN":"USA","Los Angeles":"USA","Sacramento, CA":"USA","Toronto":"Canada","Los Angeles ":"USA","New York":"USA","California":"USA","New York, NY":"USA","United States":"USA","Mumbai":"India","Manhattan, NY":"USA","Los Angeles, CA":"USA","NYC area":"USA","Washington, DC":"USA","Washington, D.C.":"USA","London": "UK","California, USA":"USA"}
    data.drop(data[(data["location"] == "304") | (data["location"] == 'ss')].index, inplace=True)
    disaster_data = data[data['target'] == 1]

    loc_disaster = disaster_data["location"].str.replace('\$\$','\\$\\$').apply(lambda x : x if (x not in location_map)  else location_map.get(x)).value_counts(sort=True, dropna=True)

    
    return location_plot(loc_disaster,10)
    # image_loc_no_disaster_polt = location_plot(loc_no_disaster)


# def pandas_profile(path):
#     pfr = pandas_profiling.ProfileReport(train_data)
#     pfr.to_file(path)

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return '<img class="col-5" src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
