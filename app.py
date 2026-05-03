import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load data + word index
vocab_size = 5000
maxlen = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
word_index = imdb.get_word_index()

# Build model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train quickly (small epochs for demo)
model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)

# Prediction function
def predict_sentiment(text):
    words = text.lower().split()
    sequence = [word_index.get(word, 2) for word in words]
    padded = pad_sequences([sequence], maxlen=maxlen)
    
    prediction = model.predict(padded)[0][0]
    return prediction

# UI
st.title("🎬 Sentiment Analysis using LSTM")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    score = predict_sentiment(user_input)
    
    st.write("Prediction score:", score)
    
    if score > 0.5:
        st.success("Positive 😊")
    else:
        st.error("Negative 😠")