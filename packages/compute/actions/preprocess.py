import ast
import re
from typing import List

import nltk
import pandas as pd

# download preprocessing assets (corpus and word lists)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def _make_new_filepath(original: str, suffix: str) -> str:
    name, ext = original.split(".")
    return f"{name}_{suffix}.{ext}"


def clean(dataset_path: str) -> str:
    """
    Applies regex-based text cleaning to the 'text' column
    for every dataset row.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'text' column which contains strings.

    Returns
    -------
    `str` The path for the clean version of the dataset in the DFS.
    """
    def _remove_unused(text: str):
        clean_data = text.lower()
        clean_data = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            " ", clean_data)
        clean_data = re.sub(r"<.*>", "", clean_data)
        clean_data = re.sub(r"@[a-zA-Z0-9_]+", "", clean_data)
        clean_data = clean_data.replace("\n", "")\
            .replace("#", "")
        return clean_data

    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str
    }

    if "train" in dataset_path:
        dtypes["target"] = int

    new_path = _make_new_filepath(dataset_path, "clean")
    df = pd.read_csv(f"/data/{dataset_path}", index_col="id", dtype=dtypes)
    df["text"] = df["text"].apply(_remove_unused)
    df.to_csv(f"/data/{new_path}")
    return new_path


def tokenize(dataset_path: str) -> str:
    """
    Creates an additional columns 'tokens' to the dataset.
    It contains a list of stemmed and lemmatized tokens.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'text' column which contains strings.

    Returns
    -------
    `str` The path for the tokenized version of the dataset in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str
    }

    if "train" in dataset_path:
        dtypes["target"] = int

    new_path = _make_new_filepath(dataset_path, "tokenized")
    df = pd.read_csv(f"/data/{dataset_path}", index_col="id", dtype=dtypes)
    df["text_stemmed"] = df["text"].apply(nltk.stem.PorterStemmer().stem)
    df["text_lemmatized"] = df["text_stemmed"].apply(
        nltk.stem.WordNetLemmatizer().lemmatize)
    df["tokens"] = df["text_lemmatized"].apply(
        nltk.tokenize.RegexpTokenizer(r'\w+').tokenize)
    df.to_csv(f"/data/{new_path}")
    return new_path


def remove_stopwords(dataset_path: str) -> str:
    """
    Applies stopwords removal to the 'tokens' colums
    for each dataset row.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'tokens' column which contains
    a list of strings.

    Returns
    -------
    `str` The path for the new version of the dataset in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "text_stemmed": str,
        "text_lemmatized": str,
        "tokens": str,  # this list will be converted by Python AST
    }

    if "train" in dataset_path:
        dtypes["target"] = int

    def _rm_stopwords(tokens: List[str]):
        return [w for w in tokens
                if w not in nltk.corpus.stopwords.words('english')]

    new_path = _make_new_filepath(dataset_path, "nostopwords")
    df = pd.read_csv(
        f"/data/{dataset_path}",
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})
    df["tokens"] = df["tokens"].apply(_rm_stopwords)
    df.to_csv(f"/data/{new_path}")
    return new_path
