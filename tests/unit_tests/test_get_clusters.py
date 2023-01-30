"""
Unit test for tesing the custom functions used for MRZ detection
"""
import random
from os import listdir
from os.path import isfile, join
import unittest
from sidekick.document.document import Document
from document_splitter.get_clusters import ClusterDetection

PATH_TO_FILES = "tests/resources/RIMAC.pdf"
documents = Document(PATH_TO_FILES).image_paths
# random.shuffle(documents)

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
