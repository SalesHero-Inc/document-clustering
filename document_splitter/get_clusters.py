"""
This script contains the cluster detection function to get most representative samples
"""
from typing import List
import logging
import pandas as pd
from document_splitter.src.get_optimal_k import (
    get_elbow_point,
    get_custom_cluster_model,
)
from document_splitter.src.get_top_samples import (
    create_dataframe,
    get_all_scores,
    get_top_samples,
)
from document_splitter.enumerators import DataType, DefaultParams

log = logging.getLogger("Log Cluster Detection")


class ClusterDetection:
    """
    Class for detecting MRZ fields from images
    """

    def __init__(self, *args, **kwargs) -> None:
        super(ClusterDetection, self).__init__(*args, **kwargs)

    def get_clustered_data(
        self,
        documents: List,
        data_distinction_type: str = DataType.TEXT.value,
        n_samples: str = DefaultParams.DEFAULT_N_SAMPLES.value,
        custom_k_value: int = DefaultParams.DEFAULT_CUSTOM_K_VALUE.value,
    ) -> pd.DataFrame:
        """
        Function to detect MRZ from image
        Args:
            documents (List): list of documents
            data_distinction_type (str): type of feature (text or image)
            based on which the images can be distinguished
            n_samples (int): number of samples from each cluster
            custom_k_value (int): custom k value to cluster at
        Returns:
            top_cluster_samples (pd.Dataframe): dataframe containing clustered data
        """

        # check data type validity
        assert data_distinction_type in [
            DataType.IMAGE.value,
            DataType.TEXT.value,
        ], f"Received unsupported type: {data_distinction_type}"

        data_df = create_dataframe(documents)
        log.info("Dataframe created\n")

        # if custom k value is not given, then we detect optimal k value using eblow plot else we use given k
        if custom_k_value == DefaultParams.DEFAULT_CUSTOM_K_VALUE.value:
            optimal_k_value, best_cluster_model, embedder_model = get_elbow_point(
                data_df, data_distinction_type
            )
            log.info(f"Elbow point detected with optimal k value as {optimal_k_value} \n")
        else:
            (
                optimal_k_value,
                best_cluster_model,
                embedder_model,
            ) = get_custom_cluster_model(data_df, data_distinction_type, custom_k_value)
            log.info(f"Cluster model fited with given k value \n")
        

        all_scores = get_all_scores(
            data_df, embedder_model, best_cluster_model, data_distinction_type
        )
        log.info("Distance for all data points from cluster centroid extracted\n")

        top_cluster_samples = get_top_samples(all_scores, optimal_k_value, n_samples)
        log.info(f"Most representative samples detected - {top_cluster_samples}\n")

        return top_cluster_samples
