import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("spam_sentiment_analysis_transformed.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# converti in minuscolo
df['text'] = df['text'].str.lower()

# dividi dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# vettorizzazione 
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#modello Naive Bayes 
model = MultinomialNB()
model.fit(X_train, y_train)

# Predizione e valutazione
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# predizione per un altro file
txt_file_path = "messaggi test.txt" 

with open(txt_file_path, "r", encoding="utf-8") as f:
    messages = [line.strip().lower() for line in f if line.strip()]

messages_vec = vectorizer.transform(messages)
predictions = model.predict(messages_vec)

for msg, pred in zip(messages, predictions):
    print(f"Message: {msg}\nPrediction: {pred}\n")



