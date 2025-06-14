import pandas as pd
import os
from itertools import combinations
from scipy.spatial import distance
from typing import TypeVar
from app.api.embeddings import (
    check_if_urls_in_collection,
    get_urls_data,
    EmbeddingType,
    EmbeddingNotFoundException
)


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_PATH = os.path.join(ROOT_PATH, "data/similarities.csv")
DVAL = TypeVar('DVAL', str, float)


def calculate_similarities(embedding_type: EmbeddingType) -> dict[str, list[DVAL]]:
    """
    Calculates cosine similarity scores between pairs of embeddings for a specified type
    and returns the results in a structured dictionary format.

    The method first checks if the given embedding type exists within a defined collection.
    If the embedding type is not found, an exception is raised. It retrieves corresponding
    embeddings, computes cosine similarity scores between every unique pair of embeddings,
    and organizes the results into a dictionary. The dictionary includes similarity scores
    and the URL pairs associated with the analyzed embeddings.

    :param embedding_type: The type of embeddings to compare within the data.
    :type embedding_type: EmbeddingType

    :return: A dictionary containing lists of similarity scores for the analyzed embedding
             type and lists of corresponding URL pairs.
    :rtype: dict[str, list[DVAL]]

    :raises EmbeddingNotFoundException: If the given embedding type is not found
                                        in the collection.
    """
    results = {"urls": [], embedding_type: []}

    if not check_if_urls_in_collection(embedding_type=embedding_type):
        raise EmbeddingNotFoundException(f"Embedding type {embedding_type} not found in collection")

    embeddings = get_urls_data(data_type="embeddings", embedding_type=embedding_type)
    for embedding1, embedding2 in combinations(embeddings, 2):
        results[embedding_type].append(round(1 - distance.cosine(embedding1[1], embedding2[1]), 4))
        results["urls"].append(f"{embedding1[0]} vs {embedding2[0]}")
    return results


def save_similarities_to_csv() -> None:
    """
    Save similarity scores computed using different embedding types to a CSV file.

    This function calculates similarity scores using two embedding types:
    OpenAI and HuggingFace. The function creates DataFrames for each,
    merges them on the shared 'urls' field, and writes the result to a CSV file
    after sorting by the OpenAI similarity scores.

    :raises ValueError: Raised if the merge validation fails due to duplicate or
        mismatched rows.
    :return: None
    """
    similarities = calculate_similarities(embedding_type="openai")
    openai = pd.DataFrame(similarities)
    similarities = calculate_similarities(embedding_type="huggingface")
    huggingface = pd.DataFrame(similarities)

    merged = openai.merge(huggingface, on="urls", how='outer', validate="one_to_one")
    merged.sort_values(by="openai", ascending=False).to_csv(FILE_PATH, index=False)


def load_similarities_from_csv() -> list[dict[str, DVAL]]:
    """
    Loads similarity data from a CSV file and converts it into a list of dictionaries.

    This function reads the content of a CSV file specified in the global `FILE_PATH`
    variable, parses it into a Pandas DataFrame, and transforms the DataFrame into a
    list of dictionaries. Each dictionary represents a row in the CSV file, with
    keys being the column names and values being the corresponding field values.

    :return: List of dictionaries where each dictionary corresponds to a row in the
        CSV file. The keys of the dictionaries are the column names in the CSV file.
    :rtype: list[dict[str, DVAL]]
    """
    return pd.read_csv(FILE_PATH).to_dict(orient="records")
