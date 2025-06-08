from fastapi import APIRouter, status
from app.api.embeddings import get_top_n
from app.api.nouns import calculate_frequencies
from app.api.similarities import calculate_similarities
from app.schemas.urls import ComparisonResponse


urls_router = APIRouter()


@urls_router.get(
    "/get-stats",
    status_code=status.HTTP_200_OK,
    response_model=ComparisonResponse
)
async def get_stats():
    """
    Handles the GET request to retrieve statistical data on URL similarities, including sorted similarity metrics,
    search results, and noun frequencies.

    :return: A dictionary containing the statistics for each URL. For each URL, the following information is included:
             - 'url1': The URL being analyzed.
             - 'similarities': A list of similarity data sorted in descending order of similarity scores.
             - 'chromadb_search': A dictionary containing query text and the top 5 search results related to the URL.
             - 'nouns': The frequency of nouns related to the URL.
    """
    frequencies = await calculate_frequencies()
    similarities = await calculate_similarities()
    texts, search_results = get_top_n()
    return {'stats':
        [
            {
                'url1': url,
                'similarities': sorted(data['similarities'], key=lambda x: x['similarity'], reverse=True),
                'chromadb_search': {
                    'query_text': texts[url],
                    'top_5_results': search_results[url],
                },
                'nouns': frequencies[url],
            }
            for url, data in similarities.items()
        ]
    }
