import os
import json
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

working_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_directory, "Notebook", "Model.keras")
tokenizer_path = os.path.join(working_directory, "Notebook", "tokenizer_word_index.json")

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open(tokenizer_path, 'r') as json_file:
        word_index = json.load(json_file)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

tokenizer = Tokenizer(num_words=5000)
tokenizer.word_index = word_index

def predict_sentiment(review):
    try:
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        return sentiment
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

st.title('Sentiment Analysis on Movie Reviews')
st.write('Enter a Movie Review to predict its sentiment (Positive/Negative).')

input_review = st.text_area('Enter the review...')

if st.button('Predict Sentiment'):
    if input_review.strip():
        sentiment = predict_sentiment(input_review)
        if sentiment:
            st.write('The Sentiment of the Review is:', sentiment)
    else:
        st.write('Please enter a valid review.')
