from collections import namedtuple
from unittest import mock

import pandas as pd

# save the unpatched versions of the mocked functions
builtin_open = open
pandas_to_csv = pd.DataFrame.to_csv


def mock_open(*args, **kwargs):
    try:
        mock_dataset = next(d for d in MOCK_DATASETS if d.path == args[0])
        return mock.mock_open(read_data=mock_dataset.content)(*args, **kwargs)
    except StopIteration:
        # we keep DFS files in the test folder to avoid root permission errors
        _args = [*args]
        _args[0] = str(args[0]).replace("/data/", "")
        return builtin_open(*_args, **kwargs)


def mock_to_csv(df, path: str):
    pandas_to_csv(df, path.replace("/data/", ""))


MockDataset = namedtuple("MockDataset", ["path", "content"])

MOCK_DATASETS = [
    MockDataset(
        path="/data/dataset_raw_http.csv",
        content='''id,keyword,location,text
1,keyword,testlocation,http://google.com
2,keyword,testlocation,https://google.com
3,keyword,testlocation,test http://google.com
'''),
    MockDataset(
        path="/data/dataset_raw_tags.csv",
        content='''id,keyword,location,text
1,keyword,testlocation,<p>text</p>
2,keyword,testlocation,test <img src="something" />
'''),
    MockDataset(
        path="/data/dataset_raw_usernames.csv",
        content='''id,keyword,location,text
1,keyword,testlocation,test @test
2,keyword,testlocation,there@there
'''),
    MockDataset(
        path="/data/dataset_raw_lemmas.csv",
        content='''id,keyword,location,text
1,keyword,testlocation,I'm up for it
2,keyword,testlocation,what are you doing
2,keyword,testlocation,it at from disaster
'''),
    MockDataset(
        path="/data/dataset_preprocessed.csv",
        content='''id,keyword,location,text,target
1,keyword,testlocation,danger storm,1
2,keyword,testlocation,cute pup,0
''')]
