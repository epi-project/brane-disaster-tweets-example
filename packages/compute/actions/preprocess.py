import nltk
import pandas as pd

# download preprocessing assets (corpus and word lists)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


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
        clean_data = clean_data.replace(r"http.* ", " ")
        clean_data = clean_data.replace(r"<.*?>", "")
        return clean_data

    new_path = f"{dataset_path}_clean"
    df = pd.read_csv(dataset_path)
    df["text"].apply(_remove_unused)
    df.to_csv(new_path)
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
    new_path = f"{dataset_path}_tokenized"
    df = pd.read_csv(dataset_path)
    df["text"].apply(nltk.stem.PorterStemmer().stem)
    df["text"].apply(nltk.stem.WordNetLemmatizer().lemmatize)
    df["tokens"] = df["text"].apply(
        nltk.tokenize.RegexpTokenizer(r'\w+').tokenize)
    df.to_csv(new_path)
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
    def _rm_stopwords(tokens: list[str]):
        return [w for w in tokens
                if w not in nltk.corpus.stopwords.words('english')]

    new_path = f"{dataset_path}_nostopwords"
    df = pd.read_csv(dataset_path)
    df["tokens"].apply(_rm_stopwords)
    df.to_csv(new_path)
    return new_path
