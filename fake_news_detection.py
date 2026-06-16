import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("fake news.csv")
true = pd.read_csv("true news.csv")

# Remove unwanted unnamed columns
fake = fake.loc[:, ~fake.columns.str.contains("^Unnamed")]
true = true.loc[:, ~true.columns.str.contains("^Unnamed")]

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Select text and label columns
X = data["text"].fillna("")
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy*100,2), "%")

# User input
news = input("\nEnter a news article: ")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("✅ Real News")
else:
     print("❌ Fake News")
     