from aiohttp import ClientSession
from bs4 import BeautifulSoup
import asyncio
import hashlib
import re


WIKIPEDIA_URLS = (
    'https://pl.wikipedia.org/wiki/ChatGPT',
    'https://pl.wikipedia.org/wiki/Retrieval-augmented_generation',
    'https://pl.wikipedia.org/wiki/PLLuM',
    'https://pl.wikipedia.org/wiki/Hugging_Face',
    'https://pl.wikipedia.org/wiki/PyTorch',
    'https://pl.wikipedia.org/wiki/TensorFlow',
    'https://pl.wikipedia.org/wiki/Kwazar',
    'https://pl.wikipedia.org/wiki/Blazar',
    'https://pl.wikipedia.org/wiki/Dysk_akrecyjny',
    'https://pl.wikipedia.org/wiki/Soczewkowanie_grawitacyjne',
    'https://pl.wikipedia.org/wiki/Ciemna_energia',
    'https://pl.wikipedia.org/wiki/Zbrojni',
    'https://pl.wikipedia.org/wiki/Muzyka_duszy',
    'https://pl.wikipedia.org/wiki/Pale_Blue_Dot',
    'https://pl.wikipedia.org/wiki/The_Blue_Marble',
    'https://pl.wikipedia.org/wiki/Hekabe',
    'https://pl.wikipedia.org/wiki/Achilles',
    'https://pl.wikipedia.org/wiki/Agamemnon',
    'https://pl.wikipedia.org/wiki/Sen_Agamemnona',
    'https://pl.wikipedia.org/wiki/Penelopa',
)


def preprocess_website_content(content: str) -> str:
    """
    :param content: The raw HTML or text content of a Wikipedia page that needs preprocessing.
    :return: A cleaned string containing the primary content of the Wikipedia page, with unnecessary sections
    and references removed.
    """
    text = 'Z Wikipedii, wolnej encyklopedii '
    idx_start = content.index(text) + len(text)

    idx_end = len(content)
    sections = (
        ' Zobacz też [ edytuj | edytuj kod ]',
        ' Przypisy [ edytuj | edytuj kod ]',
        ' Bibliografia [ edytuj | edytuj kod ]',
        ' Linki zewnętrzne [ edytuj | edytuj kod ]',
        ' p d e ',
        ' Kontrola autorytatywna ( osoba ):'
    )
    for section in sections:
        if section in content:
            idx_end = content.index(section)
            break
    return re.sub(
        r' \[ \d+ \]',
        r'',
        content[idx_start:idx_end].replace('[ edytuj | edytuj kod ] ', '')
    )


async def fetch_and_extract_text(session: ClientSession, url: str) -> tuple[str, str]:
    """
    :param session: An instance of aiohttp.ClientSession used for making asynchronous HTTP requests.
    :param url: The target URL from which to fetch and extract the textual content.
    :return: A tuple containing a hashed representation of the URL and the extracted/preprocessed text content,
    or an error message if the operation fails.
    """
    try:
        async with session.get(url) as response:
            # Ensure the HTTP request is successful
            if response.status == 200:
                html_content = await response.text()

                # Parse the HTML to extract text
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text(separator=" ", strip=True)
                text_content = preprocess_website_content(text_content)
                url_hashed = hashlib.shake_256(url.encode('utf-8')).hexdigest(8)

                return url_hashed, text_content
            else:
                return f"Failed to fetch the site. HTTP status: {response.status}"

    except Exception as e:
        return f"An error occurred: {str(e)}"


async def urls_fetcher(urls: list) -> dict[str, str]:
    """
    Fetches urls and extracts their text content.
    
    :param urls: A list of URLs to fetch content from.
    :return: A dict with url hashes as keys and relevant extracted text contents for each URL as values.
    """
    final_response = {}
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_and_extract_text(session, url))
            tasks.append(task)
        for hashed, content in await asyncio.gather(*tasks):
            final_response[hashed] = content
    return final_response


async def wikipedia_urls_fetcher():
    """
    Asynchronously fetches URLs from the WIKIPEDIA_URLS source.

    This function utilizes `urls_fetcher` to retrieve the list of URLs defined in the WIKIPEDIA_URLS constant.

    :return: A list of fetched URLs from WIKIPEDIA_URLS.
    """
    return await urls_fetcher(WIKIPEDIA_URLS)


def wikipedia_urls_hashed():
    """
    Generates a list of hashed values for URLs stored in the WIKIPEDIA_URLS constant.

    The URLs are hashed using the SHAKE-256 algorithm, and each hash is truncated to 8 characters.

    :return: A list of 8 character SHAKE-256 hash strings for each URL in WIKIPEDIA_URLS.
    """
    return [hashlib.shake_256(url.encode('utf-8')).hexdigest(8) for url in WIKIPEDIA_URLS]


def wikipedia_urls_mapping() -> dict[str, str]:
    """
    Maps hashed Wikipedia URLs to their corresponding original URLs.

    This function utilizes the `wikipedia_urls_hashed` function, which generates hashed versions of Wikipedia URLs,
    and the predefined `WIKIPEDIA_URLS` list.
    It creates a dictionary mapping each hashed URL to its original URL by zipping the two sequences together.

    :return: A dictionary where keys are hashed versions of Wikipedia URLs and values are the corresponding original URLs.
    """
    return dict(zip(wikipedia_urls_hashed(), WIKIPEDIA_URLS))
