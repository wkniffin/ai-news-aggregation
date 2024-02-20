# text_analysis.py
import openai
from config import openai_api_key
openai.api_key = openai_api_key
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import extract_base_url


def extract_key_points(aiclient, text):
    response = aiclient.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": f"Analyze the article and list the 5 key findings, limiting each to a maximum of 15 words. Present each finding as a bullet point followed by a sentiment score in parentheses, like so:\n- Finding 1: [Brief summary] (Sentiment Score: X).\nEnsure each finding is concise and the sentiment score ranges from -1 (fully negative) to 1 (fully positive), with 0 being neutral.\n\n{text}"}]
    )
    return response.choices[0].message.content.strip()


def generate_group_title(aiclient, key_points):
    prompt = "Generate a concise title that summarizes these key points:\n\n" + "\n".join(key_points)
    response = aiclient.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def create_key_points_dict(aiclient, news):
    """
        Creates a dictionary of key points extracted from a collection of news articles.
        Args:
            news (dict): A dictionary containing news articles, each with an 'id' and 'text' key.
        Returns:
            dict: A dictionary where each key is an article ID and each value is the extracted key points for that article.
    """
    key_points_dict = {}
    for article in news['news']:
        text = article['text']
        key_points = extract_key_points(aiclient, text)
        key_points_dict[article['id']] = key_points
    return key_points_dict


def calculate_pairwise_similarities(embeddings):
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.transpose(0, 1))
    return similarity_matrix


def group_embeddings_based_on_similarity(similarity_matrix, similarity_threshold=0.5):
    groups = []
    already_grouped = set()
    for i in range(similarity_matrix.size(0)):
        if i in already_grouped:
            continue
        similar_indices = torch.nonzero(similarity_matrix[i] > similarity_threshold).squeeze().tolist()
        if not isinstance(similar_indices, list):
            similar_indices = [similar_indices]
        already_grouped.update(similar_indices)
        groups.append(similar_indices)
    return groups
