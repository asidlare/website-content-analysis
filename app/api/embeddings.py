from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os
from typing import Iterator, TypeVar
from app.api.urls import (
    wikipedia_urls_mapping,
    wikipedia_urls_hashed,
    wikipedia_urls_fetcher,
)


# Represents any type for Iterator, in this case embeddings or documents
T = TypeVar("T")


def get_or_create_collection() -> Collection:
    """
    Initializes and returns an existing or newly created collection from
    a ChromaDB persistent client.
    The collection is associated with the embedding function utilizing the OpenAI API.

    :return: A ChromaDB collection object for "wikipedia-urls-content-data".
    """
    # chromadb client
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    chroma_db_path = f"{root_path}/app/chroma"
    client = chromadb.PersistentClient(path=chroma_db_path)

    # wikipedia collection
    api_key = os.environ['OPENAI_API_KEY']
    return client.get_or_create_collection(
        name="wikipedia-urls-content-data",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=api_key,
        )
    )


def check_if_urls_in_collection() -> bool:
    """
    Checks if all the URLs specified by their hashed IDs exist in the collection.

    :return: True if the number of URL IDs found in the collection matches the total number of hashed URL IDs, otherwise False.
    """
    collection = get_or_create_collection()
    urls_ids = wikipedia_urls_hashed()
    return True if len(collection.get(ids=urls_ids)["ids"]) == len(urls_ids) else False


async def create_embeddings() -> None:
    """
    Asynchronously creates embeddings from fetched URLs content and adds them to a collection.

    :return: None
    """
    fetched_urls = await wikipedia_urls_fetcher()

    collection = get_or_create_collection()
    collection.add(
        ids=list(fetched_urls.keys()),
        documents=list(fetched_urls.values()),
    )


def get_urls_data(embeddings: bool = True) -> Iterator[tuple[str, T]]:
    """
    :param embeddings: A boolean flag indicating whether to include embeddings in the result.
    :return: An iterator yielding tuples, where each tuple consists of a URL string and either embeddings
    or document content, depending on the value of the `embeddings` parameter.
    """
    url_ids = wikipedia_urls_hashed()
    collection = get_or_create_collection()
    data_to_get = {'ids': url_ids}
    if embeddings:
        data_to_get['include'] = ['embeddings']

    data = collection.get(**data_to_get)
    mapping = wikipedia_urls_mapping()
    return (
        zip([mapping[id] for id in data["ids"]], data["embeddings"]) if embeddings
        else zip([mapping[id] for id in data["ids"]], data["documents"])
    )


def get_top_n(n=5) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Fetches the top N related URLs and their respective textual representations.

    :param n: The number of top related URLs to retrieve for each input, default is 5.
    :return: A tuple containing two dictionaries:
             - The first dictionary maps URLs to their textual representations.
             - The second dictionary maps URLs to a list of their N most related URLs.
    """
    if not check_if_urls_in_collection():
        create_embeddings()

    results = {}
    texts = {}

    mapping = wikipedia_urls_mapping()
    collection = get_or_create_collection()
    for id, url in mapping.items():
        text = url.split('/')[-1].replace('_', ' ')
        texts[url] = text
        results[url] = [
            mapping[id]
            for id in collection.query(
                query_texts=[text],
                n_results=n
            )["ids"][0]
        ]
    return texts, results
