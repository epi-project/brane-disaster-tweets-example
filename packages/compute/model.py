import ast
import datetime
import pickle

import pandas as pd
from sklearn.naive_bayes import BernoulliNB


def train_model(dataset_path: str, vectors_path: str) -> str:
    """
    Trains a classifier for the disaster tweets challenge.
    The trained model is stored as a binary file in the DFS.

    Parameters
    ----------
    vectors_path: `str`
    Binary file containing the vector representation of the training tweets.

    dataset_path: `str`
    CSV file containing the preprocessed train dataset.

    Returns
    -------
    `str` The path for the trained model binary dump in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "text_stemmed": str,
        "text_lemmatized": str,
        "target": int,
    }

    dataset = pd.read_csv(
        dataset_path,
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})

    with open(vectors_path, "rb") as f:
        vectors = pickle.load(f)

    model = BernoulliNB()
    model.fit(vectors, dataset["target"])

    timestamp = datetime.datetime.utcnow()
    with open("/result/model.pickle", "wb") as f:
        pickle.dump(model, f)

    return "/result/model.pickle"


def create_submission(
    dataset_path: str, vectors_path: str, model_path: str
) -> str:
    """
    Performs the classification task on each entry in the test dataset.
    It stores the final result as a CSV in the DFS in the form:

    `id,target`

    Where target is either 0 (not a disaster) or 1 (disaster).

    Parameters
    ----------
    vectors_path: `str`
    Binary file containing the vector representation of the testing tweets.

    dataset_path: `str`
    CSV file containing the preprocessed test dataset.

    model_path: `str`
    Binary file containing the trained model.

    Returns
    -------
    `str` The path for the submission CSV.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "text_stemmed": str,
        "text_lemmatized": str,
    }

    dataset = pd.read_csv(
        dataset_path,
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})

    with open(vectors_path, "rb") as f:
        vectors = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    submission = pd.DataFrame()
    submission['id'] = dataset.index.to_list()
    submission['target'] = model.predict(vectors)

    filename = f"/result/submission.csv"
    submission.to_csv(filename, index=False)

    return filename
