import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ---------- NLTK SETUP (VERY IMPORTANT) ----------
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

load_nltk()

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


# ---------- TEXT TRANSFORMATION ----------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stop_words]
    tokens = [ps.stem(i) for i in tokens]

    return " ".join(tokens)


# ---------- LOAD MODEL & VECTORIZER ----------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# ---------- STREAMLIT UI ----------
st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")

