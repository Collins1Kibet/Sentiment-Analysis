import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

working_directory = os.path.dirname(os.path.abspath(__file__))
movie_reviews_model_path = os.path.join(working_directory, 'Notebook', 'Movie_Reviews_Model.h5')
movie_reviews_tokenizer_path = os.path.join(working_directory, 'Notebook', 'tokenizer.json')

model = load_model(movie_reviews_model_path)

with open(movie_reviews_tokenizer_path, 'r') as t:
    tokenizer_data = json.load(t)
    tokenizer = Tokenizer.from_json(tokenizer_data)

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment

st.title('Sentiment Analysis on Movie Reviews')
st.write('Enter a Movie Review to predict its sentiment (Positive/Negative).')

input_review = st.text_area('Enter the review...')

if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(input_review)
    st.write('The Sentiment of the Review is: ', sentiment)
