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

# Predizione sui primi 5 messaggi del dataset
print("\n\nPredictions for the first 5 messages in the dataset:")
for i in range(5):
    print(f"\nMessage {i+1}:")
    print(df['text'].iloc[i])
    print(f"Prediction: {predict_spam(df['text'].iloc[i])}")
    print()
