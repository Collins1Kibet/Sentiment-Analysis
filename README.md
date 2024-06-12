## Movie Reviews Sentiment Analysis

This Streamlit app performs sentiment analysis on movie reviews, classifying them as positive or negative using a pre-trained neural network model.

### Features

- User-friendly web interface
- Real-time sentiment prediction
- Supports text input for movie reviews

### Jupyter Notebook

The Jupyter notebook for model creation, training, and evaluation can be found in the `Notebook` directory. Below is a brief outline of the notebook:

#### Libraries
- numpy
- pandas
- tensorflow
- sklearn
- json

### Dataset

The dataset used for training the model is the IMDB Dataset of 50K Movie Reviews, which can be found on Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Model Creation and Training

The model is created, trained, and evaluated in a Jupyter notebook. The following steps are involved:

1. **Data Loading**: Load the dataset and preprocess the text data.
2. **Tokenization**: Convert the text data into sequences of integers.
3. **Model Building**: Build a neural network model using TensorFlow/Keras.
4. **Model Training**: Train the model on the training data.
5. **Model Evaluation**: Evaluate the model on the validation data.
6. **Save Model and Tokenizer**: Save the trained model and the tokenizer word index for later use in the Streamlit app.
