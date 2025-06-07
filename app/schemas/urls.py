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


class Comparison(BaseModel):
    """
    A model representing the comparison details between two URLs.

    Attributes:
        url1 (HttpUrl): The first URL in the comparison.
        similarities (list[Similarity]): The cosine similarities between the URLs.
        nouns (dict[str, int]): The frequency of nouns extracted from the first URL.
    """
    url1: HttpUrl = Field(..., description="First URL")
    similarities: list[Similarity] = Field(..., description="Cosine similarities between the URLs")
    nouns: dict[str, int] = Field(..., description="Nouns frequency for the first URL")


class ComparisonResponse(BaseModel):
    """
        A data model representing the response containing comparison statistics for URLs.

        Attributes:
            stats (list[Comparison]): A list of Comparison objects that hold statistical comparison data for URLs.
    """
    stats: list[Comparison] = Field(..., description='Comparison stats for urls')
