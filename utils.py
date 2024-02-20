# utils.py
from urllib.parse import urlparse
import re

def extract_base_url(long_url):
    """
        Extracts the base URL from a given long URL.
        Args:
            long_url (str): The full URL from which the base URL is to be extracted.
        Returns:
            str: The base URL extracted from the given long URL.
    """
    parsed_url = urlparse(long_url)
    return parsed_url.scheme + "://" + parsed_url.netloc


def extract_points_and_scores(article_text):
    """
        Extracts findings and their sentiment scores from article text.
        Args:
            article_text (str): The text of the article containing findings and sentiment scores.
        Returns:
            list of tuples: A list where each tuple contains a finding as a string and its sentiment score as a float.
    """
    try:
        matches = re.findall(r"- Finding \d+: (.*?) \(Sentiment Score: ([^)]+)\)", article_text)
        return [(match[0], float(match[1])) for match in matches]
    except Exception as e:
        print(str(e))
        return []