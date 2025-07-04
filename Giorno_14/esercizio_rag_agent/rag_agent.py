import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter

# Il file ha header, le colonne utili sono v1 (label) e v2 (text)
data = pd.read_csv("./Giorno_12/Esercizio_Kaggle/spam.csv", encoding="latin1", usecols=[0,1], names=["label", "text"], header=0)

# Elimina eventuali valori nulli
data = data.dropna()

# Visualizza la distribuzione delle etichette
print("Distribuzione etichette:")
print(data['label'].value_counts())

# Dividi in X e y
X = data['text']
y = data['label']

# Suddividi in train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Trasformazione testo in vettori numerici
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Addestramento modello
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predizione
y_pred = model.predict(X_test_vec)

# Valutazione
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Testiamo il modello su nuove email
nuove_email = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.",
    "Ciao, ti va di andare a pranzo insieme domani?",
    "URGENT! Your account has been compromised. Please reset your password immediately.",
    "Gentile cliente, la sua bolletta è disponibile nell'area riservata.",
    "Hai vinto una crociera gratis! Rispondi subito a questo messaggio per ricevere il premio.",
    "Reminder: Your appointment is scheduled for tomorrow at 10:00 AM.",
    "Ciao, ti allego il documento richiesto.",
    "Your PayPal account has been suspended. Click here to verify your information.",
    "Gentile utente, la password è stata modificata con successo.",
    "You have received a secure message from your bank. Log in to view.",
    "Ciao, come stai? Ti va di sentirci questa sera?",
    "Act now! Lowest prices of the year on all electronics.",
    "Gentile cliente, la sua spedizione è in arrivo.",
    "Win a brand new iPhone by entering our free contest!",
    "Il tuo ordine è stato spedito e arriverà presto.",
    "Sei stato selezionato per un premio esclusivo. Rispondi subito!",
    "Buongiorno, confermiamo la ricezione del suo pagamento.",
    "Gentile cliente, la fattura è disponibile nell'area riservata.",
    "Gentile cliente, il suo abbonamento è stato rinnovato con successo."
]

nuove_email_vec = vectorizer.transform(nuove_email)
pred_nuove = model.predict(nuove_email_vec)

# Conta quante sono spam e quante non spam (ham)
conteggio = Counter(pred_nuove)
print("\n--- Risultato nuove email ---")
print(f"Spam: {conteggio.get('spam', 0)}")
print(f"Non spam: {conteggio.get('ham', 0)}")

# Etichette attese per le nuove email (esempio, da aggiornare con le tue vere etichette)
nuove_email_label = [
    "spam", "ham", "spam", "ham", "spam", "ham", "ham", "spam", "ham", "spam",
    "ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham", "ham"
]

# Calcolo F1-score sulle nuove email
f1 = f1_score(nuove_email_label, pred_nuove, pos_label="spam", average="binary")
print(f"\nF1-score sulle nuove email: {f1:.2f}")

