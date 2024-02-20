from flask import Flask, request, render_template, jsonify
from news_api_client import search_news
from text_analysis import generate_group_title, create_key_points_dict, calculate_pairwise_similarities, \
    group_embeddings_based_on_similarity
from utils import extract_points_and_scores
from config import news_api_key, openai_api_key
import openai
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize OpenAI and SentenceTransformer outside of request handling
openai.api_key = openai_api_key
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/')
def index():
    # Render a simple form for input
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    openai.api_key = openai_api_key
    aiclient = openai.OpenAI(api_key=openai.api_key)

    search_terms = request.form['search_terms']
    print('thank you')

    news = search_news(news_api_key, text=search_terms, language="en")
    print('got the news')
    key_points_dict = create_key_points_dict(aiclient, news)
    print('got the key points')

    # Extract points and their sentiment scores
    points_and_scores = [(id, point, score) for id, text in key_points_dict.items() for point, score in
                         extract_points_and_scores(text)]
    point_texts = [point for _, point, _ in points_and_scores]
    sentiment_scores = [score for _, _, score in points_and_scores]

    # Generate embeddings for points
    embeddings = model.encode(point_texts, convert_to_tensor=True)

    # Calculate pairwise similarities and group embeddings
    similarity_matrix = calculate_pairwise_similarities(embeddings)
    group_indices = group_embeddings_based_on_similarity(similarity_matrix)
    print('grouped embeddings')

    # Initialize an empty list for results
    results = []

    # Process each group to generate titles and aggregate sentiment scores
    for indices in group_indices:
        if len(indices) < 2:
            continue
        grouped_points = [point_texts[i] for i in indices]
        grouped_scores = [sentiment_scores[i] for i in indices]
        average_sentiment_score = sum(grouped_scores) / len(grouped_scores) if grouped_scores else 0

        title = generate_group_title(aiclient, grouped_points)
        results.append({
            "title": title,
            "average_sentiment_score": f"{average_sentiment_score:.2f}",
            "key_points": [f"- {text} (Sentiment Score: {score})" for text, score in
                           zip(grouped_points, grouped_scores)]
        })

    return render_template('results.html', search_terms=search_terms, results=results)


if __name__ == "__main__":
    app.run(debug=True)
