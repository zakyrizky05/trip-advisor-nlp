import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wordninja
import tensorflow as tf
import streamlit as st
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization



def app():
    #judul text
    st.title('Exploratory Data Analysis')

    #st.image(r'C:\Users\ThinkPad\Downloads\tripadvisor-logo-png_seeklogo-222085.png', width=700)

    #text biasa/data
    st.write('## Trip Advisor Hotel Reviews')

    df = pd.read_csv('tripadvisor_hotel_reviews.csv')

    st.write(df)

    st.write('## Frequent words in review text using a word cloud.')

    wc = df['Review']

    # Start with one review:
    text = " ".join(wc.astype(str))

    # Clean text: lowercase, remove HTML-like tags and non-alphabetic characters
    text = re.sub(r'<.*?>', ' ', text)  # remove HTML tags like <br>, <div>, etc.
    text = re.sub(r'\bbr\b', ' ', text)  # remove isolated 'br'
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-alphabetic
    text = text.lower()

    # Create and generate a word cloud image:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated image:
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.figure(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # No axes for word cloud
    st.pyplot(fig)

    # Top 20 frequent words in review text using a bar plot.
    st.write('## Top 20 frequent words in review text using a bar plot.')

    # Join them into a single string
    text = " ".join(wc.astype(str))

    # Clean text: lowercase, remove HTML-like tags and non-alphabetic characters
    text = re.sub(r'<.*?>', ' ', text)  # remove HTML tags like <br>, <div>, etc.
    text = re.sub(r'\bbr\b', ' ', text)  # remove isolated 'br'
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-alphabetic
    text = re.sub(r'\bnt\b', ' ', text)  # remove isolated 'nt'
    text = text.lower()

    # Tokenize words
    words = text.split()

    # Remove common stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get the 20 most common words
    top_20 = word_counts.most_common(20)
    words, counts = zip(*top_20)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Only one figure
    bars = ax.bar(words, counts, color='skyblue')
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_title("Top 20 Most Frequent Words in Reviews")
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")

    # Add annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    st.pyplot(fig)

    # Wordcloud for negative reviews
    st.write('## Wordcloud for negative reviews.')

    # Filter only low-rated reviews (rating 1 or 2)
    low_rated_reviews = df[df['Rating'].isin([1, 2])]['Review']

    # Combine into one text string
    text = " ".join(low_rated_reviews.astype(str))

    # Clean text
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\bbr\b', ' ', text)
    text = re.sub(r'\bnt\b', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Generate word cloud
    wordcloud_low = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_low, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Top 20 frequent words in negative reviews using a bar plot.
    st.write('## Top 20 frequent words in negative reviews using a bar plot.')

    # Filter only low-rated reviews (rating 1 or 2)
    low_rated_reviews = df[df['Rating'].isin([1, 2])]['Review']

    # Join them into a single string
    text = " ".join(low_rated_reviews.astype(str))

    # Clean text
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\bbr\b', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\bnt\b', ' ', text)
    text = text.lower()

    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    top_20 = word_counts.most_common(20)
    words, counts = zip(*top_20)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(words, counts, color='skyblue')
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_title("Top 20 Most Frequent Words in Bad Reviews (1-2 Ratings)")
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")

    # Add annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    st.pyplot(fig)

    # Frequency of each rating using a bar plot.
    st.write('## Frequency of each rating using a bar plot.')

    # Count the frequency of each rating
    rating_counts = df['Rating'].value_counts().sort_index()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
    ax.set_title("Number of Reviews per Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticks([1, 2, 3, 4, 5])  # Ensure all ratings are shown
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    app()