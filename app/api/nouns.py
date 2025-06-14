import pandas as pd
import os
import spacy
import stanza
import time
from collections import Counter
from more_itertools import unzip
from typing import TypeVar
from concurrent.futures import ProcessPoolExecutor
from app.api.embeddings import (
    check_if_urls_in_collection,
    get_urls_data,
    EmbeddingNotFoundException
)


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_PATH = os.path.join(ROOT_PATH, "data/nouns.csv")
DVAL = TypeVar('DVAL', str, int)


def calculate_frequencies_per_document_stanza(url: str, document: str) -> list[dict[str, DVAL]]:
    """
    Calculate lemma frequencies for nouns and proper nouns in a document using
    Stanza natural language processing library. This function processes a
    text document, extracts relevant lemmas, computes their frequencies, and
    returns a list of dictionaries containing the URL, lemma, and its frequency.

    :param url: The URL of the document being processed.
    :type url: str
    :param document: The textual content of the document to be analyzed.
    :type document: str
    :return: A list of dictionaries with each dictionary containing the document URL, a lemma, and its frequency.
    :rtype: list[dict[str, DVAL]]
    """
    t0 = time.perf_counter()
    print(f"Processing document {url} (stanza)...")

    # Initialize Stanza pipeline
    nlp = stanza.Pipeline('pl', processors='tokenize,mwt,pos,lemma', download_method=None)

    doc = nlp(document.lower())

    # Get lemmas that match allowed POS tags and calculate frequencies
    frequencies = Counter([
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.pos in ("NOUN", "PROPN") and word.lemma
    ])

    output = [
        {"url": url, 'noun': lemma, 'stanza': freq}
        for lemma, freq in frequencies.items()
        if freq > 1
    ]
    t = time.perf_counter() - t0
    print(f"Finished processing document {url} (stanza), total time: {t:.2f}s")
    return output


def calculate_frequencies_per_document_spacy(url: str, document: str) -> list[dict[str, DVAL]]:
    """
    Calculate the frequency of nouns in a given text document using the SpaCy NLP library.

    This function processes a text document using the polski (Polish) language SpaCy model to
    identify and count the lemmatized nouns and proper nouns. It constructs a frequency
    distribution for these nouns and filters the results for those appearing more than once.
    The function outputs a list of dictionaries that include the source URL, noun, and its
    frequency in the document when analyzed using SpaCy.

    :param url: The URL or identifier associated with the text document.
    :type url: str
    :param document: The textual content of the document to analyze.
    :type document: str
    :return: A list of dictionaries where each dictionary includes the source URL, a noun, and
        its frequency in the analyzed document (using SpaCy).
    :rtype: list[dict[str, DVAL]]
    """
    t0 = time.perf_counter()
    print(f"Processing document {url} (spacy)...")

    nlp = spacy.load("pl_core_news_sm")
    doc = nlp(document.lower())
    frequencies = Counter([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN")
    ])
    output = [
        {"url": url, 'noun': lemma, 'spacy': freq}
        for lemma, freq in frequencies.items()
        if freq > 1
    ]
    t = time.perf_counter() - t0
    print(f"Finished processing document {url} (spacy), total time: {t:.2f}s")
    return output


def calculate_frequencies_and_save_to_csv() -> None:
    """
    Calculates noun frequencies in documents using two NLP models (spaCy and Stanza),
    merges the results, and saves them to a CSV file. This function ensures that the
    required embeddings are available, processes the documents in parallel, and outputs
    merged results from both models into a single CSV file. The process measures and
    prints the total time taken for computation.

    :raises EmbeddingNotFoundException: If the required embeddings are not present in
        the collection.
    :return: None
    """
    if not check_if_urls_in_collection(embedding_type="openai"):
        raise EmbeddingNotFoundException("Embedding type openai not found in collection")
    t0 = time.perf_counter()
    urls, docs = map(list, unzip(get_urls_data(data_type='documents', embedding_type="openai")))
    with ProcessPoolExecutor() as executor:
        results_spacy = executor.map(calculate_frequencies_per_document_spacy, urls, docs)
    with ProcessPoolExecutor() as executor:
        results_stanza = executor.map(calculate_frequencies_per_document_stanza, urls, docs)

    output_spacy = {"url": [], "noun": [], "spacy": []}
    for result in results_spacy:
        for item in result:
            output_spacy["url"].append(item["url"])
            output_spacy["noun"].append(item["noun"])
            output_spacy["spacy"].append(item["spacy"])
    spacy_df = pd.DataFrame(output_spacy)

    output_stanza = {"url": [], "noun": [], "stanza": []}
    for result in results_stanza:
        for item in result:
            output_stanza["url"].append(item["url"])
            output_stanza["noun"].append(item["noun"])
            output_stanza["stanza"].append(item["stanza"])
    stanza_df = pd.DataFrame(output_stanza)

    merged = stanza_df.merge(spacy_df, on=["url", "noun"], how="outer").fillna(0)
    merged.to_csv(FILE_PATH, index=False)

    t = time.perf_counter() - t0
    print(f"NLP total processing time: {t:.2f}s")


def get_frequencies():
    """
    Reads a CSV file containing frequency data, processes the data, and generates a list of
    dictionaries. Each dictionary corresponds to a unique URL in the dataset, and contains
    the URL and sorted records of frequencies per URL.

    The CSV file is expected to have the following columns:
    - `url`: The URL identifier (string).
    - `stanza`: Frequency determined by a specific method, represented as integers.
    - `spacy`: Frequency determined by another method, represented as integers.
    All records are grouped by unique URLs, and the noun frequencies are sorted by
    `stanza` and `spacy` values in descending order.

    :return: A list of dictionaries containing grouped and sorted frequency records for
        unique URLs. Each dictionary has the following structure:
        - `url`: URL string.
        - `nouns`: List of dictionaries with frequency values sorted by `stanza` and
          `spacy`.
    :rtype: list
    """
    frequencies_df = pd.read_csv(FILE_PATH)
    frequencies_df["stanza"] = frequencies_df["stanza"].astype(int)
    frequencies_df["spacy"] = frequencies_df["spacy"].astype(int)

    output = []
    for url in frequencies_df["url"].unique():
        output.append({
            "url": url,
            "nouns": frequencies_df.loc[frequencies_df["url"] == url].iloc[:, 1:].sort_values(
                by=["stanza", "spacy"], ascending=[False, False]).to_dict(orient="records")
        })
    return output
