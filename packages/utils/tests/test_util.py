
from asyncore import write
import unittest
import os
from ..run import download
from unittest import mock

builtin_open = open
file_mock = unittest.mock.Mock(write=unittest.mock.Mock())
open_mock = unittest.mock.Mock(return_value=file_mock)  
    
class TestUtils(unittest.TestCase):
    
    def test_download(self):
        repo_owner = "marinoandrea";
        repo_name = "disaster-tweets-brane";
        repo_dataset_filepath_test = "data/test.csv";

        result = download(repo_owner, repo_name, repo_dataset_filepath_test, "test.csv")
        assert(result != 0)
        

        
if __name__ == '__main__':
    unittest.main()
    