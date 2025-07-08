import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


df = pd.read_csv("spam_sentiment_analysis_transformed.csv", encoding='latin-1')
#df = df[['v1', 'v2']]
#df.columns = ['label', 'text']

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

#modello Naive Bayes e random forest
model = MultinomialNB()
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Predizione e valutazione
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)

print("Naive Bayes Classification Report:")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))

print("Random Forest Classification Report:")
print(accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(f1_score(y_test, y_pred2, average='weighted'))






