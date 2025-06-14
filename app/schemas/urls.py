from pydantic import BaseModel, Field, HttpUrl


class Similarity(BaseModel):
    """
    Represents similarity results between two URLs using different embeddings.

    This class is a model that holds values for comparing two URLs based on their
    cosine similarity scores using both OpenAI and HuggingFace embeddings.

    :ivar urls: Compared URLs, in the format 'url1 vs url2'.
    :type urls: str
    :ivar openai: Cosine similarity score calculated using OpenAI embeddings.
    :type openai: float
    :ivar huggingface: Cosine similarity score calculated using HuggingFace embeddings.
    :type huggingface: float
    """
    urls: str = Field(..., description="Compared URLs, e.g. 'url1 vs url2")
    openai: float = Field(..., description="Cosine similarity using OpenAI embeddings")
    huggingface: float = Field(..., description="Cosine similarity using HuggingFace embeddings")


class Noun(BaseModel):
    """
    Represents a model for noun information and its associated frequency values.

    This class serves as a data model for storing noun information extracted
    from website content, along with its frequency values calculated using
    different NLP tools, such as Stanza and SpaCy. It aids in normalizing and
    processing the noun data across different pipelines for further analysis.

    :ivar noun: Noun extracted from website content.
    :type noun: str
    :ivar stanza: Frequency of the noun calculated using the Stanza NLP pipeline.
    :type stanza: float
    :ivar spacy: Frequency of the noun calculated using the SpaCy NLP pipeline.
    :type spacy: float
    """
    noun: str = Field(..., description="Noun extracted from website content")
    stanza: float = Field(..., description="Frequency of nouns calculated using Stanza NLP pipeline")
    spacy: float = Field(..., description="Frequency of nouns calculated using Spacy NLP pipeline")


class Nouns(BaseModel):
    """
    Represents data regarding nouns extracted from a website's content.

    This class is intended to store and manage the data related to a specific URL, along
    with a list of nouns that are identified from the content of the website. It acts as
    a data model and provides structure for handling such information.

    :ivar url: The URL of the website being analyzed for noun extraction.
    :type url: str
    :ivar nouns: A list of nouns extracted from the content of the given website.
    :type nouns: list[Noun]
    """
    url: str = Field(..., description="Compared URL")
    nouns: list[Noun] = Field(..., description="Nouns extracted from website content")


class ChromaSearch(BaseModel):
    """
    Represents a model for performing text-based search using multiple embedding sources.

    This class is designed to be used for searching and comparing text through different embedding-based
    search mechanisms, specifically using content URLs and query texts. The search results from two major
    embedding sources—OpenAI and HuggingFace—are maintained in the form of ranked lists. This can be helpful
    for various information retrieval tasks and comparative search analysis.

    :ivar url: URL of the content that serves as the basis for comparison in text search.
    :type url: HttpUrl
    :ivar query_text: Text query that specifies the search criteria, the last part of a URL.
    :type query_text: str
    :ivar openai_top5_results: List of top 5 search results retrieved using OpenAI embeddings.
    :type openai_top5_results: list[HttpUrl]
    :ivar huggingface_top5_results: List of top 5 search results retrieved using HuggingFace embeddings.
    :type huggingface_top5_results: list[HttpUrl]
    """
    url: HttpUrl = Field(..., description="URL of content used for comparison")
    query_text: str = Field(..., description="Text to search for - last part of url")
    openai_top5_results: list[HttpUrl] = Field(..., description="Search based on OpenAI embeddings")
    huggingface_top5_results: list[HttpUrl] = Field(..., description="Search based on HuggingFace embeddings")


class FinalResponse(BaseModel):
    """
    Represents the final response comprising chroma comparisons, cosine similarities,
    and extracted nouns.

    This class is used to encapsulate the results of chroma comparisons, similarity
    calculations, and noun extraction into a structured format. It is built upon
    BaseModel and provides an organized schema for handling these data parts.

    :ivar chromadb_top5: Top 5 chroma comparisons for URLs, represented as a list of
        ChromaSearch objects.
    :ivar similarities: Cosine similarities for the URLs, represented as a list of
        Similarity objects.
    :ivar nouns: Extracted nouns for the URLs, represented as a list of Nouns objects.
    """
    chromadb_top5: list[ChromaSearch] = Field(..., description="Top5 chroma comparison for urls")
    similarities: list[Similarity] = Field(..., description='Cosine similarities for urls')
    nouns: list[Nouns] = Field(..., description="Nouns for urls")
