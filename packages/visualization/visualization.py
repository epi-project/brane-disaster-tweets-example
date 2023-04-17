import ast
import base64
import io
from typing import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid")


def keywords_word_cloud(data: pd.DataFrame, path: str, store_file=False):
    """
    Generate a picture of Keywords word cloud

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe containing the submission and test datasets.

    path: `str`
    The path where the image will be saved in the DFS.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(15, 15), facecolor=None)
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(
        data["keyword"].value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    if(store_file):
        plt.savefig(path,
                    dpi=300, bbox_inches="tight")
    return fig_to_base64(fig)


def tweets_wordcloud(data: pd.DataFrame, path: str, store_file=False):
    """
    Generate a word cloud based on the word in the tweets after
    the stop word removal and text cleaning.

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    path: `str`
    The path where the image will be saved in the DFS.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(20, 20), facecolor=None)
    wordcloud = WordCloud(background_color='white')\
        .generate_from_frequencies(
            pd.Series(' '.join(data["tokens"].apply(
                lambda x: " ".join(x))).split())
        .value_counts(sort=True, dropna=True).to_dict())
    # plot the WordCloud image
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    if(store_file):
        plt.savefig(path,
                    dpi=300, bbox_inches="tight")
    return fig_to_base64(fig)


def draw_plot(data: pd.DataFrame, n, path: str, store_file=False):
    """
    Generate a top n frequency plot based on the data.

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    n: `int`
    Number of top ranked entries to plot.

    path: `str`
    The path where the image will be saved in the DFS.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(25, 12))
    data.value_counts(sort=True).nlargest(n).plot.bar()
    plt.tight_layout(pad=0)
    plt.xticks(fontproperties='Times New Roman', size=40)
    if(store_file):
        plt.savefig(path, dpi=300, bbox_inches="tight")
    return fig_to_base64(fig)


def prediction_plot(data: pd.DataFrame, store_file=False) -> str:
    """
    Generate a pie chart based on the submission result.

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    fig = plt.figure(figsize=(25, 12))
    data["target"].value_counts(sort=True, dropna=True)\
        .plot.pie(autopct="%1.1f%%", rot=0, fontsize=30,
                  labels=['Non-Disaster', 'Disaster'],
                  wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
                  textprops={'size': 'x-large'}, startangle=90,
                  explode=(0, 0.1))
    if(store_file):
        plt.savefig("/result/prediction_plot.png",
                    dpi=300, bbox_inches="tight")
    return fig_to_base64(fig)


def keywords_profile(data: pd.DataFrame, store_file=False, n_top: int = 10):
    """
    Integrates the tweets keywords word cloud and word counts plot

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    n_top:`int`
    Number of top ranked entries to plot.

    Returns
    -------
    `Array[str]`
    The list of img HTML tag in base64 format.The first two
    images are based on all data. The 3rd and 4th images are
    based on predicted disaster data. The last two are based
    on predicted non-disaster data.
    """

    # the overall data word frequency plot and word cloud
    image_tweets_plot = draw_plot(
        data['keyword'], n_top,
        "/result/keywords_profile/keywords_plot.png", store_file)
    image_tweets_word_cloud = keywords_word_cloud(
        data, "/result/keywords_profile/keywords_word_cloud.png", store_file)

    # the disaster data word frequency plot and word cloud
    disaster_data = data[data['target'] == 1]
    image_dis_tweets_plot = draw_plot(
        disaster_data['keyword'], n_top,
        "/result/keywords_profile/disaster_keywords_plot.png", store_file)
    image_dis_tweets_word_cloud = keywords_word_cloud(
        disaster_data,
        "/result/keywords_profile/disaster_keywords_word_cloud.png", store_file)

    # the non-disaster data word frequency plot and word cloud
    non_disaster_data = data[data['target'] == 0]
    image_no_dis_tweets_plot = draw_plot(
        non_disaster_data['keyword'], n_top,
        "/result/keywords_profile/non_disaster_keywords_plot.png", store_file)
    image_no_dis_tweets_word_cloud = keywords_word_cloud(
        non_disaster_data,
        "/result/keywords_profile/non_disaster_keywords_word_cloud.png",
        store_file)
    return [
        image_tweets_plot,
        image_tweets_word_cloud,
        image_dis_tweets_plot,
        image_dis_tweets_word_cloud,
        image_no_dis_tweets_plot,
        image_no_dis_tweets_word_cloud
    ]


def tweets_profile(data: pd.DataFrame, store_file=False, n_top: int = 10):
    """
    Integrates the tweets word cloud and tweets word counts plot

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    n_top:`int`
    Number of top ranked entries to plot.

    Returns
    -------
    `Array[str]`
    The list of img HTML tag in base64 format. The first two images
    are based on all data. The  3rd and 4th images are based on
    predicted disaster data. The last two are based on predicted
    non-disaster data.
    """
    # the overall data word frequency plot and word cloud
    image_tweets_plot = draw_plot(
        pd.Series(' '.join(data["tokens"].apply(lambda x: " ".join(x)))
                  .split()), n_top,
        "/result/tweets_profile/tweets_plot.png", store_file)
    image_tweets_word_cloud = tweets_wordcloud(
        data, "/result/tweets_profile/tweets_word_cloud.png", store_file)

    # the disaster data word frequency plot and word cloud
    disaster_data = data[data['target'] == 1]
    image_dis_tweets_plot = draw_plot(
        pd.Series(' '.join(disaster_data["tokens"]
                           .apply(lambda x: " ".join(x)))
                  .split()), n_top,
        "/result/tweets_profile/disaster_tweets_plot.png", store_file)
    image_dis_tweets_word_cloud = tweets_wordcloud(
        disaster_data, "/result/tweets_profile/disaster_tweets_word_cloud.png",
        store_file)

    # the non-disaster data word frequency plot and word cloud
    non_disaster_data = data[data['target'] == 0]
    image_no_dis_tweets_plot = draw_plot(
        pd.Series(' '.join(non_disaster_data["tokens"]
                           .apply(lambda x: " ".join(x))).split()),
        n_top, "/result/tweets_profile/non_disaster_tweets_plot.png", store_file)
    image_no_dis_tweets_word_cloud = tweets_wordcloud(
        non_disaster_data,
        "/result/tweets_profile/non_disaster_tweets_word_cloud.png", store_file)
    return [
        image_tweets_plot,
        image_tweets_word_cloud,
        image_dis_tweets_plot,
        image_dis_tweets_word_cloud,
        image_no_dis_tweets_plot,
        image_no_dis_tweets_word_cloud
    ]


def location_profile(data: pd.DataFrame, store_file=False, n_top=10):
    """
    Generate a plot that the top N counts of the dataset

    Parameters
    ----------
    data: `pandas.DataFrame`
    The pandas dataframe contains the submission and test datasets.

    store_file: `bool`
    Whether the images should be saved to the DFS.

    n_top:`int`
    Number of top ranked entries to plot.

    Returns
    -------
    `str` The img HTML tag in base64 format.
    """
    location_map = {
        "Worldwide": "Earth",
        "Everywhere": "Earth",
        "United Kingdom": "UK",
        "Manchester": "UK",
        "London, England": "UK",
        "Memphis, TN": "USA",
        "Dallas, TX": "USA",
        "Oklahoma City, OK": "USA",
        "San Diego, CA": "USA",
        "Pennsylvania, USA": "USA",
        "Texas": "USA",
        "Seattle": "USA",
        "Chicago": "USA",
        "Florida": "USA",
        "NYC": "USA",
        "San Francisco": "USA",
        "Atlanta, GA": "USA",
        "San Francisco, CA": "USA",
        "New York City": "USA",
        "US": "USA",
        "Denver, Colorado": "USA",
        "Chicago, IL": "USA",
        "Nashville, TN": "USA",
        "Los Angeles": "USA",
        "Sacramento, CA": "USA",
        "Toronto": "Canada",
        "Los Angeles ": "USA",
        "New York": "USA",
        "California": "USA",
        "New York, NY": "USA",
        "United States": "USA",
        "Mumbai": "India",
        "Manhattan, NY": "USA",
        "Los Angeles, CA": "USA",
        "NYC area": "USA",
        "Washington, DC": "USA",
        "Washington, D.C.": "USA",
        "London": "UK",
        "California, USA": "USA"}
    data.drop(data[(data["location"] == "304") | (
        data["location"] == 'ss')].index, inplace=True)
    disaster_data = data[data['target'] == 1]

    loc_disaster = disaster_data["location"].str.replace('\$\$', '\\$\\$')\
        .apply(
        lambda x: x if (x not in location_map) else location_map.get(x))

    return draw_plot(loc_disaster, n_top,
                     "/result/location_profile/loc_disaster.png", store_file)


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

    img_data = base64.b64encode(img.getvalue()).decode('utf-8')

    return f'<img class="col-5" src="data:image/png;base64, {img_data}">'


def plot_bigrams_distribution(
    dataset_path: str, n_top_bigrams: int = 15, store_file=False
) -> str:
    """
    Plots the bigrams occurence distribution given a dataset
    and exports an image in the DFS.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'bigrams' column.

    n_top_bigrams: `int`
    Number of bigrams to include in the visualization in descending
    order of occurrences.

    Returns
    -------
    `str` The path for the plot image in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "target": int,
    }

    df = pd.read_csv(
        dataset_path,
        index_col="id",
        dtype=dtypes,
        converters={
            "tokens": ast.literal_eval,
            "bigrams": ast.literal_eval})

    counter = Counter([x for xs in df["bigrams"].to_list() for x in xs])
    most = counter.most_common()

    x, y = [], []
    for word, count in most[:n_top_bigrams]:
        x.append(word)
        y.append(count)
    fig = plt.figure(figsize=(25, 12))

    p = sns.barplot(x=y, y=x, color="blue", palette="pastel")
    p.set_xlabel("X-Axis", fontsize=40)
    p.set_ylabel("Y-Axis", fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.ylabel("Bi-grams")
    plt.xlabel("Occurrences")

    if(not store_file):
        return fig_to_base64(fig)
    else:
        plt.savefig(
            "/result/bigrams_distribution.png",
            dpi=300, bbox_inches="tight")
        plt.close()
        return "/result/bigrams_distribution.png"


def generate_prediction_plot(
        filepath_test_dataset: str, filepath_sub_dataset: str) -> str:
    """
    Generate a plot that the top N counts of the dataset
    and exports an image in the DFS.

    Parameters
    ----------
    filepath_test_dataset: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a test dataset.

    filepath_sub_dataset: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a prediction dataset.

    Returns
    -------
    `str` The name for the plot image in the DFS.
    """

    sub_data = pd.read_csv(filepath_test_dataset)
    test_data = pd.read_csv(filepath_sub_dataset)
    predict_data = pd.merge(sub_data, test_data, on="id")

    prediction_plot(predict_data, True)

    return "prediction_plot.png"


def generate_location_profile(dataset_path: str, n_top: int = 10) -> str:
    """
    Generate a plot that the top N Disaster Location of the dataset
    and exports an image in the DFS.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'tokens' column.

    n_top: `int`
    Number of location to include in the visualization in descending
    order of counts.

    Returns
    -------
    `str` The path for the plot image in the DFS.
    """

    data = pd.read_csv(dataset_path,
                       converters={"tokens": ast.literal_eval})
    location_profile(data, True, n_top)

    return "/result/location_profile"


def generate_tweets_profile(dataset_path: str, n_top: int = 10) -> str:
    """
    Integrates the tweets word cloud and tweets word counts plot
    and exports images in the DFS.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'tokens' column.

    n_top: `int`
    Number of words to include in the visualization in descending
    order of counts.

    Returns
    -------
    `str` The path for the plot image in the DFS.
    """

    data = pd.read_csv(dataset_path,
                       converters={"tokens": ast.literal_eval})
    tweets_profile(data, True, n_top)

    return "/result/tweets_profile"


def generate_keywords_profile(dataset_path: str, n_top: int = 10) -> str:
    """
    Integrates the tweets keywords word cloud and word counts plot
    and exports images in the DFS.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'tokens' column.

    n_top: `int`
    Number of keywords to include in the visualization in descending
    order of counts.

    Returns
    -------
    `str` The path for the plot image in the DFS.
    """

    data = pd.read_csv(dataset_path,
                       converters={"tokens": ast.literal_eval})
    keywords_profile(data, True, n_top)

    return "/result/keywords_profile"
