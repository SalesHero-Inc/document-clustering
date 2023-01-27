"""
Unit test for tesing the custom functions used for MRZ detection
"""
import random
from os import listdir
from os.path import isfile, join
import unittest
from document_splitter.get_clusters import ClusterDetection

PATH_TO_FILES = "tests/resources"
documents = [f"{PATH_TO_FILES}/{f}" for f in listdir(PATH_TO_FILES) if isfile(join(PATH_TO_FILES, f))][:100]
random.shuffle(documents)

class TestClusterDetection(unittest.TestCase):
    """
    Setting up a class to run unit tests
    """

    def test_get_clusters(self):
        """
        This function is to test cluster extraction function
        """

        clusters = ClusterDetection().get_clustered_data(documents, "text")
        print (clusters)
        assert (
            len(clusters) > 16
        ), f"Failed. Was not able to extract correct keyword representation. Received {clusters}"
