import re
import nltk
import numpy as np
import wordninja
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer


nltk_data_dir = '/tmp/nltk_data'
nltk.data.path.append(nltk_data_dir)

# Download the stopwords resource
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

loaded_model = load_model('model_lstm_2.keras')

stpwds_en = set(stopwords.words('english'))
# Manually remove 'not' from stopwords to preserve it
stpwds_en.discard('not')

lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\bbr\b', ' ', text)

    # Convert "4*" to "4-star" if it's followed by space or end of string
    text = re.sub(r'\b(\d)\*(?=\s|$)', r'\1 star', text)

    # Fix contractions: convert "ca n't" → "cant", "do n't" → "dont"
    text = re.sub(r"\b(ca|do|does|did|is|are|was|were|should|would|could|must|has|have|had|might|will|wo)\s+n['’]t\b", lambda m: m.group(1) + "nt", text)
    
    # Convert standalone n't to "not"
    text = re.sub(r"\b[nN]['’]t\b", "not", text)  # rare case like just "n't"
    text = re.sub(r"\b(\w+)n['’]t\b", r"\1nt", text)  # e.g., wasn't → wasnt

    # Remove non-alphanumeric characters except apostrophes (keep for now)
    text = re.sub(r"[^\w\s]", '', text)

    # Tokenize
    words = word_tokenize(text)

    # Split merged words using wordninja
    split_words = []
    for word in words:
        if re.fullmatch(r'(?:[1-9]|[1-9]\d)(?:st|nd|rd|th)', word):
            split_words.append(word)
        else:
            split_words.extend(wordninja.split(word))

    # Remove stopwords (except 'not') and lemmatize
    final_words = [lemmatizer.lemmatize(w) for w in split_words if w not in stpwds_en]

    return " ".join(final_words)

# Define the sentiment label function
def get_sentiment_label(rating):
    if rating >= 4:
        return 1  # Positive
    elif rating == 3:
        return 2  # Neutral
    else:
        return 0  # Negative

# Define label mapping
label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

def app():
    # Streamlit app
    st.title("TripAdvisor Sentiment Analysis")
    st.write("Enter a review text to predict its sentiment.")

    # Text input
    user_input = st.text_area("Review Text", placeholder="Type your review here, e.g., 'Stayed at Sunset Oasis for a weekend...'")

    # Predict button
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.error("Please enter a review text.")
        else:
            try:
                # Prepare input as a list (model expects a list of texts)
                new_texts = [user_input]
                
                # Make prediction
                predicted_labels = np.argmax(loaded_model.predict(new_texts), axis=1)
                
                # Get sentiment label
                sentiment = label_map[predicted_labels[0]]
                
                # Display result
                st.success(f"Predicted Sentiment: **{sentiment}**")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Please ensure the model is loaded correctly and the input is valid.")

    # Instructions
    st.markdown("---")
    st.write("**Instructions**: Enter a review text and click 'Predict Sentiment' to see the sentiment (Positive, Neutral, or Negative).")