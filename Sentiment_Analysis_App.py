import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequense import pad_sequences

model = load_model('Movie_Reviews_Model.h5')

with open('tokenizer.json', 'r') as t:
    tokenizer_data = json.load(t)
    tokenizer = Tokenizer.from_json(tokenizer_data)

def predict_sentiment(review):
    sequence = tokenizer.text_to_sequence([review])
    padded_sequence = pad_sequences(sequence, maxlen= 200)
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment

st.title('Sentiment Analysis on Movie Reviews')
st.write('Enter a Movie Review to predict its sentiment (Positive/Negative).')

input_review = st.text_area('Enter the review...')

if st.button('Predict Sentiment'):
    sentiment = predict_sentiment(input_review)
    st.write('The Sentiment of the Review is: ', sentiment)