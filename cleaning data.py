import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
df = pd.read_csv("news.csv")
df['label'] = df['label'].astype(str).str.strip().str.upper()
stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words and len(word) > 2])
df['clean_text'] = df['text'].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
print("TF-IDF Matrix Shape:", X.shape)
print(df[['label', 'clean_text']].head())
