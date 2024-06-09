import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the working directory and paths to the model and tokenizer
working_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_directory, "Notebook", "Movie_Reviews_Model.h5")
tokenizer_path = os.path.join(working_directory, "Notebook", "tokenizer_word_index.json")

# Load the pre-trained model
model = load_model(model_path)

# Load the tokenizer's word index from the JSON file
with open(tokenizer_path, 'r') as json_file:
    word_index = json.load(json_file)

# Initialize and configure the tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.word_index = word_index

def predict_sentiment(review):
    # Convert the review to sequences
    sequence = tokenizer.texts_to_sequences([review])
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=200)
    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment

# Streamlit UI
st.title('Sentiment Analysis on Movie Reviews')
st.write('Enter a Movie Review to predict its sentiment (Positive/Negative).')

input_review = st.text_area('Enter the review...')

if st.button('Predict Sentiment'):
    if input_review.strip():
        sentiment = predict_sentiment(input_review)
        st.write('The Sentiment of the Review is: ', sentiment)
    else:
        st.write('Please enter a valid review.')
