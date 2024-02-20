# news_api_client.py
import requests

def search_news(api_key, text=None, source_countries=None, language=None,
                min_sentiment=None, max_sentiment=None, earliest_publish_date=None,
                latest_publish_date=None, news_sources=None, authors=None, entities=None,
                location_filter=None, sort=None, sort_direction=None, offset=None, number=None):
    """
        Searches for news articles based on the provided parameters.
        Returns:
            dict: A JSON object containing the search results if the request is successful.
    """

    base_url = "https://api.worldnewsapi.com/search-news"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    params = {
        "text": text,
        "source-countries": source_countries,
        "language": language,
        "min-sentiment": min_sentiment,
        "max-sentiment": max_sentiment,
        "earliest-publish-date": earliest_publish_date,
        "latest-publish-date": latest_publish_date,
        "news-sources": news_sources,
        "authors": authors,
        "entities": entities,
        "location-filter": location_filter,
        "sort": sort,
        "sort-direction": sort_direction,
        "offset": offset,
        "number": number
    }
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()