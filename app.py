import streamlit as st

st.title("🎬 Sentiment Analysis using LSTM")

# Simple keyword-based fallback model (for deployment)
positive_words = ["love", "amazing", "great", "fantastic", "good", "excellent"]
negative_words = ["hate", "terrible", "bad", "boring", "worst"]

def predict_sentiment(text):
    text = text.lower()
    
    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1
    
    return score

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    score = predict_sentiment(user_input)
    
    st.write("Score:", score)
    
    if score > 0:
        st.success("Positive 😊")
    elif score < 0:
        st.error("Negative 😠")
    else:
        st.warning("Neutral 😐")
