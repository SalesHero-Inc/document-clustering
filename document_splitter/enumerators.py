"""Lists enumerators used in the scope of document splitting"""

from enum import Enum


class EncoderModel(Enum):
    """Enumerates huggingface encoder models"""

    TEXT = "sentence-transformers/distilroberta-base-msmarco-v2"
    IMAGE = "sentence-transformers/clip-ViT-B-32"


class DataType(Enum):
    """Enumerates scope"""

    TEXT = "text"
    IMAGE = "image"


class ColumnNames(Enum):
    """Enumerates column names in the dataset"""

    FULL_TEXT = "full_text"
    IMAGE_NAME = "image_name"
    KEYWORD_REPRESENTATION = "keyword_rep"
    CLUSTER_REPRESENTATION = "cluster_representation"
    CLUSTER = "cluster"
    SCORE = "score"
