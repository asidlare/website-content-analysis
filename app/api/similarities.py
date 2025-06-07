from collections import defaultdict
from itertools import combinations
from scipy.spatial import distance
from app.api.embeddings import check_if_urls_in_collection, create_embeddings, get_urls_data


async def calculate_similarities():
    """
    Asynchronously calculates the cosine similarity between all pairs of embeddings obtained from URLs.

    If embeddings for the URLs are not already present in the collection, it creates them first.
    The function iterates over each pair of embeddings, calculates the cosine similarity, and stores the results
    in a dictionary.
    Each key in the dictionary corresponds to a URL, associated with a list of similarity results
    when compared with other URLs.

    :return:
        A dictionary where each key is a URL, and the value is a dictionary containing a list of similarities with other URLs,
        sorted in descending order based on the number of similarity entries per URL.
    """
    if not check_if_urls_in_collection():
        await create_embeddings()

    results = defaultdict(lambda: {'similarities': []})

    embeddings = get_urls_data(embeddings=True)
    for embedding1, embedding2 in combinations(embeddings, 2):
        similarity = 1 - distance.cosine(embedding1[1], embedding2[1])
        results[embedding1[0]]['similarities'].append({
            'url2': embedding2[0],
            'similarity': round(float(similarity), 4)
        })
    return dict(sorted(results.items(), key=lambda x: len(x[1]['similarities']), reverse=True))
