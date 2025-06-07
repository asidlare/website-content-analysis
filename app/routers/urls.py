from fastapi import APIRouter, status
from app.api.similarities import calculate_similarities
from app.api.nouns import calculate_frequencies
from app.schemas.urls import ComparisonResponse


urls_router = APIRouter()


@urls_router.get(
    "/get-stats",
    status_code=status.HTTP_200_OK,
    response_model=ComparisonResponse
)
async def get_stats():
    """
    Fetches and returns statistical data related to URL comparisons.

    This endpoint calculates similarities and frequencies for a set of URLs. The results include the following details
    for each URL:
    - Similarities: A sorted list of similar items based on a similarity score in descending order.
    - Nouns: Frequency of occurrence for specific nouns.

    :return: A dictionary containing comparison statistics, including URL-specific similarities and noun frequencies.
    """
    similarities = await calculate_similarities()
    frequencies = await calculate_frequencies()
    return {'stats':
        [
            {
                'url1': url,
                'similarities': sorted(data['similarities'], key=lambda x: x['similarity'], reverse=True),
                'nouns': frequencies[url],
            }
            for url, data in similarities.items()
        ]
    }
