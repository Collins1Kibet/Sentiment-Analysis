import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Defining the working directory and paths to the model and tokenizer
working_directory = os.path.dirname(os.path.abspath(__file__))

movie_reviews_model_path = os.path.join(working_directory, 'Notebook', 'Movie_Reviews_Model.h5')
tokenizer_config_path = os.path.join(working_directory, 'Notebook', 'tokenizer_config.json')
tokenizer_word_index_path = os.path.join(working_directory, 'Notebook', 'tokenizer_word_index.json')

# Ensuring the paths are correct
assert os.path.isfile(movie_reviews_model_path), "Model file not found."
assert os.path.isfile(tokenizer_config_path), "Tokenizer config file not found."
assert os.path.isfile(tokenizer_word_index_path), "Tokenizer word index file not found."

try:
    model = load_model(movie_reviews_model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")

try:
    with open(tokenizer_config_path, 'r') as json_file:
        tokenizer_config = json.load(json_file)
    tokenizer = Tokenizer.from_config(tokenizer_config)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

try:
    with open(tokenizer_word_index_path, 'r') as json_file:
        word_index = json.load(json_file)
    tokenizer.word_index = word_index
    print("Tokenizer word index loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer word index: {e}")

def predict_sentiment(review):
    try:
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        return sentiment
    except Exception as e:
        return f"Error predicting sentiment: {e}"

st.title('Sentiment Analysis on Movie Reviews')
st.write('Enter a Movie Review to predict its sentiment (Positive/Negative).')

input_review = st.text_area('Enter the review...')

if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(input_review)
    st.write('The Sentiment of the Review is: ', sentiment)
