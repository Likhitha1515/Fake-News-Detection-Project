import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("fake news.csv")
true = pd.read_csv("true news.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)
x = data["text"]
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# Select text and label columns
X = data["text"]
y = data["label"]

print(data.columns)
# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")# Train model
news = input("\nEnter a news article: ")

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)
print("prediction value:",prediction[0])

if prediction[0] == 1:
    print("✅ Real News")
else:
    print("❌ Fake News")
     