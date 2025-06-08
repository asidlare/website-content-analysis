from pydantic import BaseModel, Field, HttpUrl


class Similarity(BaseModel):
    """
    Represents the similarity details between two URLs.

    Attributes:
        url2 (HttpUrl): Compared URL.
        similarity (float): Cosine similarity between two URLs.
    """
    url2: HttpUrl = Field(..., description="Compared URL")
    similarity: float = Field(..., description="Cosine similarity between two URLs")


class ChromadbSearch(BaseModel):
    """
    A model class used for structuring the data related to search functionality in a ChromaDB system.

    Attributes:
        query_text: A string representing the text to search for.
        top_5_results: A list of URLs representing the top 5 search results.
    """
    query_text: str = Field(..., description="Text to search for")
    top_5_results: list[HttpUrl] = Field(..., description="Top 5 results")


class Comparison(BaseModel):
    """
    Represents a comparison model for analyzing similarities and differences between two URLs.

    Attributes:
        url1 (HttpUrl):
            First URL to be compared.
        chromadb_search (ChromadbSearch):
            Chromadb search results containing relevant data for comparison.
        similarities (list[Similarity]):
            A list of cosine similarity metrics between the URLs.
        nouns (dict[str, int]):
            A dictionary representing the frequency of nouns extracted from the first URL.
    """
    url1: HttpUrl = Field(..., description="First URL")
    chromadb_search: ChromadbSearch = Field(..., description="Chromadb search results")
    similarities: list[Similarity] = Field(..., description="Cosine similarities between the URLs")
    nouns: dict[str, int] = Field(..., description="Nouns frequency for the first URL")


class ComparisonResponse(BaseModel):
    """
        A data model representing the response containing comparison statistics for URLs.

        Attributes:
            stats (list[Comparison]): A list of Comparison objects that hold statistical comparison data for URLs.
    """
    stats: list[Comparison] = Field(..., description='Comparison stats for urls')
