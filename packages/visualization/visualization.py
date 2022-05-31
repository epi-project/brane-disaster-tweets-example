#!/usr/bin/python3

# get_ipython().system('pip install pandas numpy seaborn matplotlib missingno wordcloud nltk pandas-profiling[notebook] lxml')
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

def keywords_word_cloud(data):
    """
    Generate a picture of Keywords word cloud

    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize = (15, 15), facecolor = None)
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(data["keyword"].value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image                      
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)


def remove_stopwords(text:str):
    """
    Applies stopwords removal to the text.

    Parameters
    ----------
    text: `str`
    The content which is needed to apply stopwords removal.

    Returns
    -------
    `str` The text that after stopwords removal.
    """
    words = [w for w in text if w not in stopwords.words('english')]
    return  ' '.join(words)

def clean_text(text:str):
    """
    Applies regex-based text cleaning to the 'text'.

    Parameters
    ----------
    text: `str`
    The content which is needed to apply regex-based text cleaning.

    Returns
    -------
    `str` The text after text cleaning.
    """
    clean_data = text.lower()
    clean_data = re.sub(r"http.*", " ",clean_data)
    clean_data = re.sub(r"[0-9]+", " ",clean_data)
    clean_data = re.sub(r"%20", " ",clean_data)
    clean_data = re.sub(r"_", " ",clean_data)
    clean_data = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ",clean_data)
    return clean_data

def tweets_wordcloud(data):
    """
    Generate a word cloud based on the word in the tweets after the stop word removal and text cleaning.

    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize = (20, 20), facecolor = None)
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(pd.Series(' '.join(data["token_data"]).split()).value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image                      
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    return fig_to_base64(fig)

def draw_plot(data,n):
    """
    Generate a top n frequency plot based on the data.
    
    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.
    n:`int`
    It controls return top n rank data.
    
    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(25,12))
    data.value_counts(sort=True).nlargest(n).plot.bar()
    plt.tight_layout(pad = 0)
    plt.xticks(fontproperties = 'Times New Roman', size = 40)
    return fig_to_base64(fig)


def prediction_plot(data):
    """
    Generate a pie chart based on the submission result.
    
    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.
    
    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(25,12))
    data["target"].value_counts(sort=True, dropna=True).plot.pie(autopct="%1.1f%%",rot=0 , fontsize=30,labels=['Non-Disaster','Disaster'],wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},textprops={'size': 'x-large'},startangle=90,explode=(0, 0.1))
    return fig_to_base64(fig)

def keywords_profile(data):
    """
    Integrates the tweets keywords word cloud and word counts plot

    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.

    Returns
    -------
    `Array[str]` The list of img HTML tag in base64 format.The first two images are based on all data. The  3rd and 4th images are based on predicted disaster data. The last two are based on predicted non-disaster data.
    """

    #the overall data word frequency plot and word cloud
    image_tweets_polt = draw_plot(data['keyword'],10)
    image_tweets_word_cloud = keywords_word_cloud(data)
    
    #the disaster data word frequency plot and word cloud
    disaster_data = data[data['target'] == 1]
    image_dis_tweets_polt = draw_plot(disaster_data['keyword'],10)
    image_dis_tweets_word_cloud = keywords_word_cloud(disaster_data)
    
    #the non-disaster data word frequency plot and word cloud
    non_disaster_data = data[data['target'] == 0]
    image_no_dis_tweets_polt = draw_plot(non_disaster_data['keyword'],10)
    image_no_dis_tweets_word_cloud = keywords_word_cloud(non_disaster_data)
    return [image_tweets_polt,image_tweets_word_cloud,image_dis_tweets_polt,image_dis_tweets_word_cloud,image_no_dis_tweets_polt,image_no_dis_tweets_word_cloud]

def tweets_profile(data):
    """
    Integrates the tweets word cloud and tweets word counts plot

    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.

    Returns
    -------
    `Array[str]` The list of img HTML tag in base64 format. The first two images are based on all data. The  3rd and 4th images are based on predicted disaster data. The last two are based on predicted non-disaster data.
    """
    data["clean_data"] = data["text"].apply(clean_text)
    tokenizer = RegexpTokenizer(r'\w+')
    data["token_data"] = data["clean_data"].apply(tokenizer.tokenize)
    data["token_data"] = data["token_data"].apply(remove_stopwords)
    #the overall data word frequency plot and word cloud
    image_tweets_polt = draw_plot(pd.Series(' '.join(data["token_data"]).split()),10)
    image_tweets_word_cloud = tweets_wordcloud(data)
    
    #the disaster data word frequency plot and word cloud
    disaster_data = data[data['target'] == 1]
    image_dis_tweets_polt = draw_plot( pd.Series(' '.join(disaster_data["token_data"]).split()),10)
    image_dis_tweets_word_cloud = tweets_wordcloud(disaster_data)
    
    #the non-disaster data word frequency plot and word cloud
    non_disaster_data = data[data['target'] == 0]
    image_no_dis_tweets_polt = draw_plot( pd.Series(' '.join(non_disaster_data["token_data"]).split()),10)
    image_no_dis_tweets_word_cloud = tweets_wordcloud(non_disaster_data)
    return [image_tweets_polt,image_tweets_word_cloud,image_dis_tweets_polt,image_dis_tweets_word_cloud,image_no_dis_tweets_polt,image_no_dis_tweets_word_cloud]
    
    
def location_profile(data):
    """
    Generate a plot that the top 10 counts of the dataset
    
    Parameters
    ----------
    data: `dataframe`
    The pandas dataframe contains the submission and test datasets.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    location_map = {"Worldwide":"Earth","Everywhere":"Earth","United Kingdom":"UK","Manchester":"UK","London, England":"UK","Memphis, TN":"USA","Dallas, TX":"USA","Oklahoma City, OK":"USA","San Diego, CA":"USA","Pennsylvania, USA":"USA","Texas":"USA","Seattle":"USA","Chicago":"USA","Florida":"USA","NYC":"USA","San Francisco":"USA","Atlanta, GA":"USA","San Francisco, CA":"USA","New York City":"USA","US":"USA","Denver, Colorado":"USA","Chicago, IL":"USA","Nashville, TN":"USA","Los Angeles":"USA","Sacramento, CA":"USA","Toronto":"Canada","Los Angeles ":"USA","New York":"USA","California":"USA","New York, NY":"USA","United States":"USA","Mumbai":"India","Manhattan, NY":"USA","Los Angeles, CA":"USA","NYC area":"USA","Washington, DC":"USA","Washington, D.C.":"USA","London": "UK","California, USA":"USA"}
    data.drop(data[(data["location"] == "304") | (data["location"] == 'ss')].index, inplace=True)
    disaster_data = data[data['target'] == 1]

    loc_disaster = disaster_data["location"].str.replace('\$\$','\\$\\$').apply(lambda x : x if (x not in location_map)  else location_map.get(x))

    return draw_plot(loc_disaster,10)


# def pandas_profile(path):
#     pfr = pandas_profiling.ProfileReport(train_data)
#     pfr.to_file(path)

def fig_to_base64(fig):
    """
    Convert the figure to base64 string

    Parameters
    ----------
    fig: `matplotlib.figure`

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return '<img class="col-5" src="data:image/png;base64, {}">'.format(base64.b64encode(img.getvalue()).decode('utf-8'))
