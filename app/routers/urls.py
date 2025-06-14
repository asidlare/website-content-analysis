from fastapi import APIRouter, status
from app.api.embeddings import get_top5
from app.api.nouns import get_frequencies
from app.api.similarities import load_similarities_from_csv
from app.schemas.urls import FinalResponse


urls_router = APIRouter()


@urls_router.get(
    "/get-stats",
    status_code=status.HTTP_200_OK,
    response_model=FinalResponse
)
async def get_stats():
    """
    Handles the retrieval of statistical data used for analysis purposes.
    The function collects and processes top 5 ChromaDB results, similarity
    metrics from a CSV file, and word frequency data.

    :return: A dictionary containing three keys: `chromadb_top5`, `similarities`,
        and `nouns`. Each key holds the corresponding processed data.
    :rtype: dict
    """
    return {
        "chromadb_top5": get_top5(),
        "similarities": load_similarities_from_csv(),
        "nouns": get_frequencies()
    }
