import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


data = {
    'text': [
        "I love this product, it's amazing!",
        "Worst experience ever, totally disappointed",
        "It was okay, nothing special",
        "Absolutely fantastic service",
        "I hate it, very bad experience",
        "Neutral opinion about this",
        "Great quality and fast delivery",
        "Terrible support, never buying again",
        "Average experience, could be better",
        "Loved it, will buy again!"
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'positive'
    ]
}

df = pd.DataFrame(data)

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

df['clean_text'] = df['text'].apply(preprocess)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()
