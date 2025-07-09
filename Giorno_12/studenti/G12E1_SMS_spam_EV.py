import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"C:\Users\MF579CW\OneDrive - EY\Desktop\spam.csv", encoding='latin1')

#preprocessing
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df=df.dropna(subset=['message'])
df['message']= df['message'].str.lower()

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorization and model training
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print()
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(model, 'sms_classifier.joblib')
joblib.dump(vectorizer, 'sms_vectorizer.joblib')




#prova messaggi test.txt
with open(r"C:\Users\MF579CW\OneDrive - EY\Desktop\EY_scripts\eai-academy\Giorno_12\esercizi\messaggi test.txt", "r", encoding="utf-8") as f:
    nuovi_messaggi = [riga.strip().replace('"', '') for riga in f if riga.strip()]

model = joblib.load('sms_classifier.joblib')
vectorizer = joblib.load('sms_vectorizer.joblib')

X_nuovi = vectorizer.transform(nuovi_messaggi)
predizioni = model.predict(X_nuovi)

for messaggio, label in zip(nuovi_messaggi, predizioni):
    print(f"{messaggio} è classificato {label}\n")
