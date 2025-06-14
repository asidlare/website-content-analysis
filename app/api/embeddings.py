from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction
)
import chromadb
import os
from typing import Iterator, Literal, TypeVar
from app.api.urls import (
    wikipedia_urls_mapping,
    wikipedia_urls_hashed,
    wikipedia_urls_fetcher,
)


class EmbeddingTypeNotRecognizedException(Exception):
    pass


class EmbeddingNotFoundException(Exception):
    pass


EmbeddingType = Literal["openai", "huggingface"]
DataType = Literal["embeddings", "documents", "similarities", "nouns"]
# Represents any type for Iterator, in this case embeddings or documents
T = TypeVar("T")


def get_or_create_collection(embedding_type: EmbeddingType = "openai") -> Collection:
    """
    Retrieve or create a chroma database collection for storing embeddings based on
    the specified embedding type. This function initializes a persistent chromadb
    client, identifies the target embedding type, and generates a corresponding
    embedding function to register with the relevant collection. If the specified
    embedding type is not recognized, an exception will be raised.

    :param embedding_type: The type of embedding function to use. Accepted values are
        "openai" or "huggingface". Defaults to "openai".
    :type embedding_type: EmbeddingType
    :return: The initialized or retrieved chromadb collection object corresponding
        to the embedding type.
    :rtype: Collection
    :raises EmbeddingTypeNotRecognizedException: If the provided ``embedding_type``
        is not recognized.
    """
    # chromadb client
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    chroma_db_path = f"{root_path}/app/chroma"
    client = chromadb.PersistentClient(path=chroma_db_path)

    if embedding_type == "openai":
        # wikipedia collection
        api_key = os.environ['OPENAI_API_KEY']
        return client.get_or_create_collection(
            name="wikipedia-urls-content-data-openai",
            embedding_function=OpenAIEmbeddingFunction(
                model_name="text-embedding-3-small",
                api_key=api_key,
            )
        )
    elif embedding_type == "huggingface":
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
        return client.get_or_create_collection(
            name="wikipedia-urls-content-data-bert",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        )
    else:
        raise EmbeddingTypeNotRecognizedException(f"Embedding type {embedding_type} not recognized.")


def check_if_urls_in_collection(embedding_type: EmbeddingType) -> bool:
    """
    Check if URLs in the collection exist.

    This function validates whether all hashed Wikipedia URLs are present
    within a specific collection identified by the embedding type.

    :param embedding_type: The type of embedding associated with the collection
        used for validation.
    :type embedding_type: EmbeddingType
    :return: A boolean indicating whether all hashed Wikipedia URLs exist
        in the collection.
    :rtype: bool
    """
    collection = get_or_create_collection(embedding_type=embedding_type)
    urls_ids = wikipedia_urls_hashed()
    return True if len(collection.get(ids=urls_ids)["ids"]) == len(urls_ids) else False


async def create_embeddings() -> None:
    """
    Creates embeddings for fetched Wikipedia URLs using specified embedding types.

    This function first retrieves URLs from the Wikipedia URL fetcher, then processes
    them using different embedding types. For each embedding type, a corresponding
    collection is either retrieved or created. The URLs and their associated document
    texts are then added to the respective collections.

    :raises KeyError: If the required keys are missing during URL fetching.

    :return: This function is asynchronous and does not return any value.
    """
    fetched_urls = await wikipedia_urls_fetcher()

    for embedding_type in ("openai", "huggingface"):
        collection = get_or_create_collection(embedding_type=embedding_type)
        collection.add(
            ids=list(fetched_urls.keys()),
            documents=list(fetched_urls.values()),
        )


def get_urls_data(
        data_type: DataType = "embeddings",
        embedding_type: EmbeddingType = "openai"
) -> Iterator[tuple[str, T]]:
    """
    Fetches and returns data paired with their corresponding Wikipedia URLs in an iterator format.

    This function retrieves data from a specified collection based on `data_type` and `embedding_type`.
    The collection is used to fetch items that match unique URL identifiers, enabling the generation
    of an iterator of tuples containing Wikipedia URL mappings along with their associated data
    (embeddings or documents).

    :param data_type: Specifies the type of data to retrieve. Possible values are "embeddings"
                      (default) or "documents". Determines if embeddings or documents are fetched.
    :type data_type: DataType
    :param embedding_type: Specifies the type of embedding to use when interacting with the
                           collection. Defaults to "openai".
    :type embedding_type: EmbeddingType
    :return: An iterator of tuples, where each tuple consists of a Wikipedia URL
             and associated data (either embeddings or documents), based on the requested data type.
    :rtype: Iterator[tuple[str, T]]
    """
    url_ids = wikipedia_urls_hashed()
    collection = get_or_create_collection(embedding_type=embedding_type)
    data_to_get = {'ids': url_ids}
    if data_type == "embeddings":
        data_to_get['include'] = ['embeddings']

    data = collection.get(**data_to_get)
    mapping = wikipedia_urls_mapping()
    match data_type:
        case "embeddings":
            return zip([mapping[id] for id in data["ids"]], data["embeddings"])
        case "documents":
            return zip([mapping[id] for id in data["ids"]], data["documents"])


def get_top5():
    """
    Retrieves the top 5 results for text similarity queries using OpenAI and Huggingface
    embeddings from collections. The collections are queried for the top matches for
    each query text, and the results are mapped back to their respective URLs.

    Raises an exception if the required embedding types are not found within the
    collections. Processes a predefined mapping of Wikipedia URLs to generate query
    text based on the URL format.

    :raises EmbeddingNotFoundException: If the required embedding type is not found in
        the collection.
    :return: A list of dictionaries where each dictionary contains the original URL,
        its derived query text, and the top 5 results for both OpenAI and Huggingface
        embedding collections.
    :rtype: list[dict]
    """
    for embedding_type in ("openai", "huggingface"):
        if not check_if_urls_in_collection(embedding_type=embedding_type):
            raise EmbeddingNotFoundException(f"Embedding type {embedding_type} not found in collection.")

    results = []

    mapping = wikipedia_urls_mapping()
    collection_openai = get_or_create_collection(embedding_type="openai")
    collection_huggingface = get_or_create_collection(embedding_type="huggingface")
    for id, url in mapping.items():
        text = url.split('/')[-1].replace('_', ' ')
        results.append({
            "url": url,
            "query_text": text,
            "openai_top5_results": [
                mapping[id]
                for id in collection_openai.query(
                    query_texts=[text],
                    n_results=5
                )["ids"][0]
            ],
            "huggingface_top5_results": [
                mapping[id]
                for id in collection_huggingface.query(
                    query_texts=[text],
                    n_results=5
                )["ids"][0]
            ],
        })
    return results
