"""
This script contains the data creation and distance based scoring functions
"""
import logging
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from sidekick.paddle_engine import PaddleEngine

from document_splitter.enumerators import ColumnNames, DataType
from document_splitter.src.get_optimal_k import get_keywords

OCR_ENGINE = PaddleEngine(
    use_angle_cls=True, lang="en", enable_mkldnn=True, rec_batch_num=16
)
log = logging.getLogger("Log sample detection")


def create_dataframe(documents: List) -> pd.DataFrame:
    """
    Function to create dataframe from images.
    If data_distinction_type is image, dataframe
    will contain just the image names.
    If data_distinction_type is text, dataframe
    will contain full text from OCR and keyword representation.
    Args:
        documents (List): list of documents
    Returns:
        data_df (pd.DataFrame): dataframe containing image names
        along with text if text distinctive feature

    """

    data_df = pd.DataFrame(
        columns=[
            ColumnNames.IMAGE_NAME.value,
            ColumnNames.PAGE_NUM.value,
            ColumnNames.FULL_TEXT.value,
            ColumnNames.KEYWORD_REPRESENTATION.value,
        ]
    )

    for df_index, image_name in enumerate(documents):
        data_df.at[df_index, ColumnNames.IMAGE_NAME.value] = image_name

        image = Image.open(image_name)

        # extract full text from image
        full_text, _ = OCR_ENGINE.image_to_text(image)
        log.info(f"Full text extracted is - {full_text} \n")

        # extract keyword representation from full text
        page_keywords = get_keywords(full_text)
        log.info(f"Keyword representation extracted is - {page_keywords} \n")

        data_df.at[df_index, ColumnNames.KEYWORD_REPRESENTATION.value] = page_keywords
        data_df.at[df_index, ColumnNames.FULL_TEXT.value] = full_text
        data_df.at[df_index, ColumnNames.PAGE_NUM.value] = df_index + 1

    return data_df


def get_individual_embedding(
    data_df: pd.DataFrame, data_distinction_type: str, i: int, embedder_model
) -> np.array:
    """
    Function to get embedding of individual data point
    Args:
        data_df (pd.DataFrame): dataframe containing all samples
        data_distinction_type (str): type of feature (text or image)
        i (int): index of row
        embedder_model: the embedder model based on image distinctive feature
    Returns:
        embedded_data (np.array): embedded data as a dataframe
    """
    if data_distinction_type == DataType.IMAGE.value:
        embedded_data = embedder_model.encode(
            Image.open(data_df.at[i, ColumnNames.IMAGE_NAME.value])
        )
    else:
        embedded_data = embedder_model.encode(
            data_df.at[i, ColumnNames.FULL_TEXT.value]
        )

    return embedded_data


def get_all_scores(
    data_df: pd.DataFrame,
    embedder_model,
    best_cluster_model,
    data_distinction_type: DataType,
) -> pd.DataFrame:
    """
    Function to calculate distance of all points from their cluster centroids
    and get the smallest one distance from their own cluster centroid.
    Args:
        data_df (pd.DataFrame): dataframe containing the images
        embedder_model: the embedder model based on image distinctive feature
        best_cluster_model: cluster model trained with optimal K value
        data_distinction_type (str): type of feature (text or image)
        based on which the images can be distinguished
    Returns:
        scores (pd.DataFrame): cluster distance scores for all samples

    """

    # check data type
    assert data_distinction_type in [
        DataType.TEXT.value,
        DataType.IMAGE.value,
    ], f"Received unsupported type: {data_distinction_type}"

    scores = pd.DataFrame(columns=data_df.columns.tolist())
    for i in range(0, len(data_df)):
        scores.at[i, ColumnNames.IMAGE_NAME.value] = data_df.at[
            i, ColumnNames.IMAGE_NAME.value
        ]
        scores.at[i, ColumnNames.PAGE_NUM.value] = data_df.at[
            i, ColumnNames.PAGE_NUM.value
        ]

        # document encoding depends on whether data_distinction_type is image or text.
        embedded_data = get_individual_embedding(
            data_df, data_distinction_type, i, embedder_model
        )
        scores.at[i, ColumnNames.FULL_TEXT.value] = data_df.at[
            i, ColumnNames.FULL_TEXT.value
        ]
        scores.at[i, ColumnNames.KEYWORD_REPRESENTATION.value] = data_df.at[
            i, ColumnNames.KEYWORD_REPRESENTATION.value
        ]

        embedded_data = embedded_data.reshape(1, -1)
        # respective distances of each point from all cluster centroids
        x_dist = best_cluster_model.transform(embedded_data) ** 2

        scores.at[i, ColumnNames.CLUSTER.value] = best_cluster_model.predict(
            embedded_data
        )[0]
        # get smallest distance as distance of point from it's respective centroid
        scores.at[i, ColumnNames.SCORE.value] = np.min(x_dist)
        log.info(
            f"Distance of point from own cluster centroid is - {scores.at[i, ColumnNames.SCORE.value]} \n"
        )

    return scores


def get_top_samples(
    scores: pd.DataFrame, optimal_k_value: int, n_samples: int
) -> pd.DataFrame:
    """
    Function to get most represenative samples of each cluster by
    getting least distant points of each cluster from their centroid.
    Args:
        scores (pd.DataFrame): cluster distance scores for all samples
        optimal_k_value (int): optimal K value
        n_samples (int): number of samples from each cluster
    Returns:
        top_samples (pd.DataFrame): top n samples from each cluster
    """
    top_samples = pd.DataFrame(
        columns=scores.columns.tolist().extend(
            [ColumnNames.CLUSTER_REPRESENTATION.value]
        )
    )

    for i in range(0, optimal_k_value):
        scores_i = scores[scores[ColumnNames.CLUSTER.value].isin([i])]
        scores_i = scores_i.sort_values(by=[ColumnNames.SCORE.value])

        # get top n samples
        scores_i = scores_i[:n_samples]

        # get cluster representation by using keyword extraction
        cluster_keywords_all = " ".join(
            scores_i[ColumnNames.KEYWORD_REPRESENTATION.value].tolist()
        )
        cluster_representation = get_keywords(cluster_keywords_all)
        scores_i[ColumnNames.CLUSTER_REPRESENTATION.value] = cluster_representation
        scores_i[ColumnNames.WITHIN_CLUSTER_INDEX.value] = list(
            np.arange(1, len(scores_i) + 1)
        )

        top_samples = pd.concat([top_samples, scores_i], axis=0)

    log.info("Most representative samples of each cluster detected  \n")

    return top_samples
