import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# load the imdb datasets words index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# Load trained model
model = tf.keras.models.load_model("model.h5")

# step-2 Helper function to decode the reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#function to preprocess user input
def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

# step-3
#prediction function
def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction=model.predict(preprocessed_input)

  sentiment='positive' if prediction[0][0] >0.5 else 'Negative'

  return sentiment,prediction[0][0]


# Streamlit UI
st.title("IMDb Sentiment Analysis 🎬")
st.write("Enter a movie review to predict sentiment.")

user_input = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed_text = preprocess_text(user_input)
        prediction = model.predict(processed_text)[0][0]

        if prediction > 0.5:
            st.success(f"Positive Review 😊 (Confidence: {prediction:.2f})")
        else:
            st.error(f"Negative Review 😞 (Confidence: {1 - prediction:.2f})")
