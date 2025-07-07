import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopword_set = set(stopwords.words('english'))
lemmatizer_nltk = WordNetLemmatizer()

# Preprocessing function
def clean_text(txt):
    txt = txt.lower()
    tokens = word_tokenize(txt)
    tokens = [t for t in tokens if t not in stopword_set]
    tokens = [lemmatizer_nltk.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df_spam = pd.read_csv('"C:\Users\MF579CW\OneDrive - EY\Desktop\spam.csv"', encoding='latin-1')
df_spam = df_spam.iloc[:, :2]
df_spam.columns = ['target', 'text']

df_spam['target'] = df_spam['target'].map({'ham': 0, 'spam': 1})
df_spam['text'] = df_spam['text'].apply(clean_text)

print(df_spam.target.value_counts())
print(df_spam.isnull().sum())

# Suddivisione in training e test set
X_tr, X_te, y_tr, y_te = train_test_split(
    df_spam['text'],
    df_spam['target'],
    test_size=0.1,
    random_state=42,
    stratify=df_spam['target']
)

vectorizer = CountVectorizer()
X_tr_vec = vectorizer.fit_transform(X_tr)
X_te_vec = vectorizer.transform(X_te)

clf = LogisticRegression()
clf.fit(X_tr_vec, y_tr)

y_predicted = clf.predict(X_te_vec)
acc = accuracy_score(y_te, y_predicted)
print(f'Accuracy: {acc:.2f}')
print(classification_report(y_te, y_predicted, target_names=['ham', 'spam']))

# nuovi messaggi:
with open(r"C:\Users\MF579CW\OneDrive - EY\Desktop\EY_scripts\eai-academy\Giorno_12\esercizi\messaggi test.txt", "r", encoding="utf-8") as file_in:
    nuovi_testi = [linea.strip().replace('"', '') for linea in file_in if linea.strip()]

nuovi_testi_proc = [clean_text(m) for m in nuovi_testi]
nuovi_testi_vec = vectorizer.transform(nuovi_testi_proc)
nuove_pred = clf.predict(nuovi_testi_vec)

print("\nNew Messages Predictions:")
for testo, pred in zip(nuovi_testi, nuove_pred):
    etichetta = 'spam' if pred == 1 else 'ham'
    print(f'[{etichetta}] Message: "{testo}"')

solo_ham = df_spam[df_spam['target'] == 0].copy()

modello_sent = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer_sent = AutoTokenizer.from_pretrained(modello_sent)
modello_hf = AutoModelForSequenceClassification.from_pretrained(modello_sent)

#sentiment analysis
pipe_sent = pipeline("sentiment-analysis", model=modello_hf, tokenizer=tokenizer_sent)

mappa_label = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

solo_ham["sentiment"] = solo_ham["text"].apply(lambda x: mappa_label[pipe_sent(x)[0]["label"]])
print(solo_ham["sentiment"].value_counts())