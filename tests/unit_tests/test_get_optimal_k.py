"""
Unit test for tesing the custom functions used for MRZ detection
"""
import json
import unittest
from document_splitter.src.get_optimal_k import get_keywords

with open("tests/test_config.json") as json_file:
    config = json.load(json_file)

class TestClusterDetection(unittest.TestCase):
    """
    Setting up a class to run unit tests
    """

    def test_get_keywords(self):
        """
        This function is to test keyword extraction function
        """
        for i in range(1, 4):
            text = config[str(i)]
            keyword_representation = get_keywords(text)

            assert (
                len(keyword_representation) > 25
            ), f"Failed. Was not able to extract correct keyword representation. Received {keyword_representation} from text - {text}"
