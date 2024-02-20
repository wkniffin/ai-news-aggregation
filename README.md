# GPT News Aggregator
Most recently, I have been using OpenAI with Python to create a news aggregation and sentiment tool to help users quickly understand the news and the biases of current news topics. This requires the ability to find news articles from a diverse range of publications relevant to the user's input. The articles are then analyzed by gpt-3.5-turbo to extract the 5 key findings of each article, with emphasis on conciseness. Here is the prompt used for extracting the key findings:

```python
"content": f"Analyze the article and list the 5 key findings, limiting each to a maximum of 15 words. Present each finding as a bullet point followed by a sentiment score in parentheses, like so:\n- Finding 1: [Brief summary] (Sentiment Score: X).\nEnsure each finding is concise and the sentiment score ranges from -1 (fully negative) to 1 (fully positive), with 0 being neutral.\n\n{text}"

```

I then use SentanceTransformers (https://www.sbert.net/) to group the key findings across all the articles into findings that refer to the same overarching idea. I take those groups and generate a title using gpt-3.5-turbo again, here's the prompt:

```python
prompt = "Generate a concise title that summarizes these key points:\n\n" + "\n".join(key_points)
```

The results are then displayed to the user with each group title above the key points related to that topic. Here are some screenshots of the *extemely basic* web app UI:

[![search terms](https://ibb.co/fr84PNb "search terms")](https://ibb.co/fr84PNb "search terms")

[![results](https://ibb.co/KFxpZcD "results")](https://ibb.co/KFxpZcD "results")
