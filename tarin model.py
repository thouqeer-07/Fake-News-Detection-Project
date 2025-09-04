import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
df = pd.read_csv("news.csv")
df['combined'] = df['title'].fillna('') + " " + df['text'].fillna('')
X = df['combined']
y = df['label']
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_vect = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("Classification Report:\n", classification_report(y_test, log_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
best_model = rf_model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nLabel distribution:")
print(df['label'].value_counts())
try:
    from google.colab import files
    files.download("model.pkl")
    files.download("vectorizer.pkl")
except:
    print("Files ready: model.pkl and vectorizer.pkl (download manually if not in Colab)")
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
title = "Kerry to go to Paris in gesture of sympathy"
text = "U.S. Secretary of State John Kerry will visit Paris to express sympathy after attacks."

combined_input = title + " " + text
input_vector = vectorizer.transform([combined_input])
prediction = model.predict(input_vector)
print("Prediction:", prediction[0])

import pickle
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
title = "Kerry to go to Paris in gesture of sympathy"
text = ""
if title and not text:
    combined_input = title
elif text and not title:
    combined_input = text
elif title and text:
    combined_input = title + " " + text
else:
    raise ValueError("Please provide at least title or text for prediction.")
input_vector = vectorizer.transform([combined_input])
prediction = model.predict(input_vector)

print("Prediction:", prediction[0])
