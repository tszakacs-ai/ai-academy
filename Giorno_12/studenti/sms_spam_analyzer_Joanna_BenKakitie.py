import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, confusion_matrix

# Caricamento del dataset
df = pd.read_csv("Giorno_12\studenti\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocessa i dati
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['text'] = df['text'].str.lower()

# Train e test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vettorizzazione del testo
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Addestramento del modello
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Valutazione del modello
y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Funzione di predizione
def predict_spam(message):
    message = message.lower()
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return "Spam" if prediction[0] == 1 else "Ham"
<<<<<<< HEAD

# Predizione sui primi 5 messaggi del dataset
print("\n\nPredictions for the first 5 messages in the dataset:")
for i in range(5):
    print(f"\nMessage {i+1}:")
    print(df['text'].iloc[i])
    print(f"Prediction: {predict_spam(df['text'].iloc[i])}")
    print()
=======

# Predizione sui primi 5 messaggi del dataset
print("\n\nPredictions for the first 5 messages in the dataset:")
for i in range(5):
    print(f"\nMessage {i+1}:")
    print(df['text'].iloc[i])
    print(f"Prediction: {predict_spam(df['text'].iloc[i])}")
    print()


print("===================================================")

# ===================================================
# Fine-tuning del modello con nuovi messaggi
# ===================================================

# Aggiunta di nuovi messaggi per testare la funzione di predizione
new_messages = pd.read_csv("Giorno_12\studenti\messaggi_aggiuntivi.csv", encoding='latin-1', sep=',')
new_messages.columns = ['label', 'text']

print(f"\n\nPercentage of new messages added: {len(new_messages) / len(df) * 100:.2f}%")

print("\n\nPredictions for first 5 new messages:")
for i in range(5):
    print(f"\nNew Message {i+1}:")
    print(new_messages['text'].iloc[i])
    print(f"Prediction: {predict_spam(new_messages['text'].iloc[i])}")
    print()

print("===================================================")


# ===================================================
# Sentiment analysis
# ===================================================
import os, certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from transformers import pipeline

# Merge dei dataset
df_all = pd.concat([df, new_messages], ignore_index=True)

# Inizializzazione pipeline Hugging Face
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Analisi del sentiment
def classify_sentiment(text):
    try:
        result = classifier(text[:512])[0]  # Troncamento per evitare input troppo lunghi
        return result['label']  # POSITIVE o NEGATIVE
    except Exception as e:
        return "ERROR"

# Applicazione del modello
print("Sto eseguendo la classificazione... (potrebbe richiedere qualche minuto)")
df_all['sentiment'] = df_all['text'].apply(classify_sentiment)

print("\n\nSentiment Analysis Results:")
print(df_all[['text', 'sentiment']].head(10))

print("===================================================")
>>>>>>> features/joanna-benkakitie
