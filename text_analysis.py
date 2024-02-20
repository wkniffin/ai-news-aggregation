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


def extract_key_facts_test(aiclient, text):
    user_message = "Please analyze the provided article according to the instructions.\n\n" + text

    response = aiclient.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "Review the text of the article provided and analyze each significant claim. Do not include claims of opinion. For each claim, you should: \n1) Summarize the claim in no more than 15 words.\n2) Provide a succinct fact-check summary in no more than 20 words. If you do not have enough info, then create a Google Search query string to find the results.\n3) Assign a truthfulness score from 0.0 to 1.0, where 0.0 is entirely false and 1.0 is entirely true. \n4) Include the source of the fact-check, if available.\n\nFormat your response as follows:\n- **Claim [n]**: [Summarize the claim here in 15 words or less] ... **Fact score**: [0.0-1.0]\n- **Fact-Check**: [Provide a summary of the fact-check here in 25 words or less]\n- **Source**: [Cited source or Google Search query string]\n\nPlease follow these additional rules:\n1) Do not include any additional qualification or explanation of your results. You must only include the information in the requested format, and nothing else. \n2) Fact-checks must be verified with a source other than the article itself, unless the article provides direct evidence.\n3) Only provide a Google Search query string if you do not have enough information to fact-check without additional sources.\n3) Google Search query strings must be as precise as possible."
            },
            {
                "role": "user",
                "content": user_message
            },
        ],
        temperature=0.5,
        top_p=0.95,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )
    return response, response.choices[0].message.content.strip()

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


def create_key_points_dict_2(aiclient, news):
    key_points_dict = {}

    def extract_key_points_for_article(article):
        text = article['text']
        key_points = extract_key_points(aiclient, text)
        base_url = extract_base_url(article['url'])  # Extract the base URL
        return article['id'], key_points, base_url  # Return the base URL along with other info

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_article = {executor.submit(extract_key_points_for_article, article): article for article in news['news']}
        for future in as_completed(future_to_article):
            article_id, key_points, base_url = future.result()
            key_points_dict[article_id] = {"key_points": key_points, "base_url": base_url}  # Store key points and base URL

    return key_points_dict


def extract_key_points_for_article(aiclient, article):
    text = article['text']
    key_points = extract_key_points(aiclient, text)
    base_url = extract_base_url(article['url'])  # Extract the base URL
    return article['id'], key_points, base_url


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
