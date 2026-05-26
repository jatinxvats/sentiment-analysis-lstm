import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Saved Model + Vocabulary
# -----------------------------

model = load_model("sentiment_lstm_model.h5")

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

maxlen = 200

# -----------------------------
# Preprocessing Function
# -----------------------------

def preprocess_text(text):

    # Lowercase + remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    words = text.split()

    # Convert words to integers
    sequence = [word_index.get(word, 2) for word in words]

    # Pad sequence
    padded = pad_sequences([sequence], maxlen=maxlen)

    return padded

# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(text):

    processed_text = preprocess_text(text)

    prediction = model.predict(processed_text)[0][0]

    confidence = float(prediction) * 100

    if prediction > 0.5:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😠"

    return sentiment, confidence

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Sentiment Analysis using LSTM",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Sentiment Analysis using LSTM")

st.markdown("""
This application uses a trained **LSTM (Long Short-Term Memory)** neural network  
to classify text sentiment as **Positive** or **Negative**.
""")

st.write("---")

# Input Box
user_input = st.text_area(
    "Enter your review:",
    height=150
)

# Analyze Button
if st.button("Analyze Sentiment"):

    if len(user_input.strip()) == 0:

        st.warning("⚠️ Please enter some text.")

    else:

        sentiment, confidence = predict_sentiment(user_input)

        st.subheader(f"Prediction: {sentiment}")

        st.write(f"Confidence Score: {confidence:.2f}%")

        # Confidence bar
        st.progress(min(int(confidence), 100))

# -----------------------------
# Sidebar Information
# -----------------------------

st.sidebar.title("📌 Model Information")

st.sidebar.markdown("""
### Architecture
- Embedding Layer
- LSTM Layer
- Dense Output Layer

### Dataset
- IMDB Movie Reviews

### Model Accuracy
- ~85% Test Accuracy

### NLP Pipeline
Text → Tokenization → Padding → LSTM → Prediction
""")
>>>>>>> 2fa1ba86c2075274530f572174786359f4469f42
