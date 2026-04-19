import pickle
import random
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vectorizer.pkl", "rb"))
lemmatizer = WordNetLemmatizer()


MIN_PRED_LEN = 40
MIN_RANDOM_LEN = 80





def preprocess(text: str):
    text = text.strip().lower()
    if not text:
        return None

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]

    if not cleaned:
        return None

    transformed = vector.transform([" ".join(cleaned)]).toarray()
    return transformed




st.title("Welcome to VerText")
st.caption("Fast check to gauge whether text is AI-generated.")

text = st.text_area("Text input", placeholder="Type or paste your text here...", height=200)

col_analyze, col_clear = st.columns([2, 1])

with col_analyze:
    analyze_clicked = st.button("Analyze")



if analyze_clicked:
    if len(text.strip()) < MIN_PRED_LEN:
        st.warning(f"Please enter at least {MIN_PRED_LEN} characters for analysis.")
        st.stop()

    features = preprocess(text)

    if features is None:
        st.warning("Please enter enough meaningful text to analyze.")
    else:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        result_col, score_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.success("Human-written")
            elif prediction == 2:
                st.error("Likely Mixed-Text")
            else:
                st.error("Likely AI - Generated")

        st.metric("Confidence Score", f"{confidence:.1f}%")
