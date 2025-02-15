import os
import streamlit as st
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import praw
from googleapiclient.discovery import build
from textblob import TextBlob
from prophet import Prophet
from sklearn.cluster import KMeans
from pytrends.request import TrendReq
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# OpenRouter API Key
API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize APIs
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIPY_CLIENT_SECRET')))

reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent='artist-marketing-ai/1.0'
)

youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
pytrends = TrendReq(hl='en-US', tz=360)


# Sentiment Analysis Function
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity


# Prophet Model for Engagement Forecasting
def predict_fan_engagement(spotify_data):
    if spotify_data.empty:
        return pd.DataFrame()

    spotify_data['ds'] = pd.to_datetime(spotify_data['release_date'])
    spotify_data['y'] = spotify_data['popularity']

    model = Prophet()
    model.fit(spotify_data[['ds', 'y']])
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat']]


# K-Means for Fan Segmentation
def cluster_fan_engagement():
    data = np.array([[500, 2000, 50], [300, 1500, 20], [700, 3000, 100]])
    kmeans = KMeans(n_clusters=3)
    return kmeans.fit_predict(data)


# AI Marketing Chatbot using OpenRouter
import requests

# API_KEY = "your_openrouter_api_key_here"  # Replace with your actual API key


def get_artist_profile(artist_name):
    """Fetch Spotify and social data for the artist."""
    # Example: Get Spotify Data
    results = sp.search(q=artist_name, type='artist', limit=1)
    if results['artists']['items']:
        artist = results['artists']['items'][0]
        followers = artist['followers']['total']
        popularity = artist['popularity']
        genres = ", ".join(artist['genres'])

        return f"Artist Name: {artist_name}, Followers: {followers}, Popularity: {popularity}, Genres: {genres}"

    return "Artist data not found."


# # Load API Key from .env file
# load_dotenv()
#
# API_KEY = os.getenv("TOGETHER_API_KEY")
#
# if not API_KEY:
#     raise ValueError("🚨 ERROR: Missing Together AI API Key! Set 'TOGETHER_API_KEY' in your .env file.")
#
#
# def get_marketing_advice(query, artist_name):
#     url = "https://api.together.xyz/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
#
#     artist_data = f"Generating marketing insights for {artist_name}..."
#
#     data = {
#         "model": "mistralai/Mistral-7B-Instruct-v0.1",  # ✅ FREE Together AI model
#         "messages": [
#             {"role": "system", "content": "You are a global artist marketing expert."},
#             {"role": "system", "content": f"Here is the artist's profile:\n{artist_data}"},
#             {"role": "user", "content": query}
#         ]
#     }
#
#     response = requests.post(url, json=data, headers=headers)
#
#     try:
#         response_json = response.json()
#         if response.status_code == 200 and "choices" in response_json:
#             return response_json["choices"][0]["message"]["content"]
#         else:
#             return f"API Error: {response_json.get('error', 'Unknown error')}"
#     except requests.exceptions.JSONDecodeError:
#         return "Error: Failed to decode JSON response from API."
def get_marketing_advice(query):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",  # ✅ Free model
        "messages": [
            {"role": "system", "content": "You are a K-Pop marketing expert."},
            {"role": "user", "content": query}
        ]
    }

    response = requests.post(url, json=data, headers=headers)

    try:
        response_json = response.json()
        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response_json.get('error', 'Unknown error')}"
    except requests.exceptions.JSONDecodeError:
        return "Error: Failed to decode JSON response from API."

# Streamlit Dashboard
st.set_page_config(page_title="AI Marketing Bot", layout="wide")
st.title("Global Artist Marketing AI")

artist_name = st.text_input("Enter Artist Name:", "JIMIN")

# Fetch Spotify Data
st.header("Spotify Data Analysis")
results = sp.search(q=artist_name, type='artist', limit=1)
if results['artists']['items']:
    artist_id = results['artists']['items'][0]['id']
    top_tracks = sp.artist_top_tracks(artist_id)['tracks']
    spotify_data = pd.DataFrame([{
        'name': track['name'],
        'popularity': track['popularity'],
        'release_date': track['album']['release_date']
    } for track in top_tracks])
    st.dataframe(spotify_data)

    # Predict Engagement Trends
    st.subheader("Fan Engagement Prediction")
    forecast = predict_fan_engagement(spotify_data)
    if not forecast.empty:
        st.line_chart(forecast.set_index("ds"))

# Fetch Reddit Sentiment Analysis
st.header("Reddit Sentiment Analysis")
submissions = reddit.subreddit("all").search(artist_name, limit=5)
comments = []
for submission in submissions:
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        if isinstance(comment, praw.models.Comment):
            comments.append({"body": comment.body, "sentiment": analyze_sentiment(comment.body)})

if comments:
    sentiment_df = pd.DataFrame(comments)
    sentiment_df["sentiment_category"] = sentiment_df["sentiment"].apply(
        lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    st.dataframe(sentiment_df)

# AI Marketing Chatbot
st.header("AI Marketing Strategy Chatbot")
user_query = st.text_area("Ask the AI for marketing insights:", "How can I increase my YouTube engagement?")
if st.button("Get Advice"):
    advice = get_marketing_advice(user_query)
    st.success(advice)