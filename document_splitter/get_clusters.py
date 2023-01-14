"""
This script contains the cluster detection function to get most representative samples
"""
from typing import List
import logging
import pandas as pd
from document_splitter.src.get_optimal_k import get_elbow_point
from document_splitter.src.get_top_samples import (
    create_dataframe,
    get_all_scores,
    get_top_samples,
)

log = logging.getLogger("Log Cluster Detection")


class ClusterDetection:
    """
    Class for detecting MRZ fields from images
    """

    def __init__(self, *args, **kwargs) -> None:
        super(ClusterDetection, self).__init__(*args, **kwargs)

    def get_clustered_data(
        self, documents: List, data_distinction_type="text"
    ) -> pd.DataFrame:
        """
        Function to detect MRZ from image
        Args:
            documents (List): list of documents
            data_distinction_type (str): type of feature (text or image)
            based on which the images can be distinguished
        Returns:
            top_cluster_samples (pd.Dataframe): dataframe containing clustered data
        """
        data_df = create_dataframe(documents)
        log.info("Dataframe created\n")

        optimal_k_value, best_cluster_model, embedder_model = get_elbow_point(
            data_df, data_distinction_type
        )
        log.info(f"Elbow point detected with optimal k value as {optimal_k_value} \n")

        all_scores = get_all_scores(
            data_df, embedder_model, best_cluster_model, data_distinction_type
        )
        log.info("Distance for all data points from cluster centroid extracted\n")

        top_cluster_samples = get_top_samples(all_scores, optimal_k_value)
        log.info("Most representative samples detected\n")
        print (top_cluster_samples)
        
        return top_cluster_samples
