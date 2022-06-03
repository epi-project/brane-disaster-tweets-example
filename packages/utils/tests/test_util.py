import os
import unittest
from unittest import mock

from ..run import download

# save the unpatched versions of the mocked functions
builtin_open = open


def mock_open(*args, **kwargs):
    _args = [*args]
    _args[0] = str(args[0]).replace("/data/", "")
    return builtin_open(*_args, **kwargs)


class TestUtils(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        os.remove("test.csv")

    @mock.patch("builtins.open", mock_open)
    def test_download(self):
        repo_owner = "marinoandrea"
        repo_name = "disaster-tweets-brane"
        repo_dataset_filepath_test = "data/test.csv"

        result = download(repo_owner, repo_name,
                          repo_dataset_filepath_test, "test.csv")
        assert(result == 0)
        assert(os.path.exists("test.csv"))


if __name__ == '__main__':
    unittest.main()
