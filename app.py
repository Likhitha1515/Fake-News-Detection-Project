import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("fake news.csv")
true = pd.read_csv("true news.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# Prepare data
X = data["text"]
y = data["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Website UI
st.title("📰 Fake News Detection System")

news = st.text_area("Enter a news article")

if st.button("Predict"):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")