import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned = ' '.join([word for word in words if word not in stop_words and len(word) > 2])
    return cleaned

st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article snippet to check whether it's **FAKE** or **REAL**.")

user_input = st.text_area("📝 Paste News Text Here:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        st.subheader("🔎 Prediction Result:")
        if str(prediction).lower() in ['fake', '1']:
            st.error("🟥 This news is likely **FAKE**.")
        else:
            st.success("🟩 This news is likely **REAL**.")
