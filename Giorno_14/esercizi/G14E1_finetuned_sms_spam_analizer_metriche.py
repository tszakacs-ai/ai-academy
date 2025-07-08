import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
import time

# Caricamento del dataset
DATASET_PATH = "SMSSpamCollection" # Il dataset originale con prestazioni non sufficienti
# DATASET_PATH = "SMSSpamCollectionFinetuning" # Fine-tuning per categorizzare meglio le truffe nuove
data = pd.read_csv(DATASET_PATH, sep="\t", header=None, names=["label", "message"])

# Preprocessing delle etichette
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Preprocessing dei messaggi usando CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Creazione del modello Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Addestramento con barra di progresso
print("Inizio addestramento Random Forest...")
start_time = time.time()
model.fit(X_train_vec, y_train)
end_time = time.time()
print(f"Addestramento completato in {end_time - start_time:.2f} secondi.\n")

# Valutazione del modello
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred, digits = 4)
print(f"Accuracy sul test set: {accuracy:.2f}\n")

precision = precision_score(y_test, y_pred, digits = 4)
print(f"Precision sul test set: {precision:.2f}\n")


# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# Test su messaggi di esempio
test_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT! Your account has been suspended. Click here to resolve immediately.",
    "Can you send me the report by 5 PM today?",
    "WINNER! Call 555-123-4567 to claim your prize. Limited time only!",
    "Are you free this weekend for the party?",
    "You have been pre-approved for a $5000 loan. Apply now!",
    "Happy birthday! Hope you have a great day!",
    "Claim your free iPhone 14 now! Offer expires soon.",
    "Don't forget about the team meeting at 3 PM."
]

# Preprocessing e predizione
test_vec = vectorizer.transform(test_messages)
test_predictions = model.predict_proba(test_vec)

# Mostra risultati con confidenza minima del 0.01%
MIN_CONFIDENCE = 0.01

for i, msg in enumerate(test_messages):
    spam_confidence = max(test_predictions[i][1] * 100, MIN_CONFIDENCE)  # Probabilità di Spam
    ham_confidence = max((1 - test_predictions[i][1]) * 100, MIN_CONFIDENCE)  # Probabilità di Ham

    label = "Spam" if spam_confidence > 50 else "Ham"
    confidence = spam_confidence if label == "Spam" else ham_confidence

    print(f"Messaggio: {msg}")
    print(f"  Predizione: {label} ({confidence:.2f}% confidenza)")
