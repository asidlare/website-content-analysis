import spacy
import time
from collections import Counter
from more_itertools import unzip
from concurrent.futures import ProcessPoolExecutor
from app.api.embeddings import check_if_urls_in_collection, create_embeddings, get_urls_data


def calculate_frequencies_per_document(url: str, document: str) -> tuple[str, dict[str, int]]:
    """
    :param url: The URL identifying the document being processed.
    :param document: The textual content of the document to process.
    :return: A tuple containing the document's URL and a dictionary of noun and proper noun lemmas with their frequencies,
    filtered to include only those with a frequency greater than 1.
    """
    t0 = time.perf_counter()
    print(f"Processing document {url}...")

    nlp = spacy.load("pl_core_news_sm")
    doc = nlp(document.lower())
    counter = Counter([
        token.lemma_
        for token in doc
        if token.pos_ in ("NOUN", "PROPN")
    ])
    sorted_dict = {
        key: value
        for key, value in sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if value > 1
    }
    t = time.perf_counter() - t0
    print(f"Finished processing document {url}, total time: {t:.2f}s")
    return url, sorted_dict


async def calculate_frequencies() -> dict[str, dict[str, int]]:
    """
    Asynchronously calculates word frequencies for a collection of documents.

    This function first verifies if the URLs exist in the collection. If not, it creates embeddings for the documents.
    Then, using a `ProcessPoolExecutor`, it processes the documents in parallel to calculate word frequencies
    for each document.

    The computation time for the NLP processing is measured and printed.

    :return: A dictionary where the keys are URLs of the documents and the values are dictionaries containing word frequencies.
    """
    if not check_if_urls_in_collection():
        await create_embeddings()
    t0 = time.perf_counter()
    urls, docs = map(list, unzip(get_urls_data(embeddings=False)))
    with ProcessPoolExecutor() as executor:
        results = executor.map(calculate_frequencies_per_document, urls, docs)
    t = time.perf_counter() - t0
    print(f"NLP total processing time: {t:.2f}s")
    return dict(results)
