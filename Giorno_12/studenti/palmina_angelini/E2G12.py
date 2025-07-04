import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # per lemmatizzazione

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)


spam_dataset = pd.read_csv('Giorno_12\studenti\palmina_angelini\spam.csv', encoding = 'latin-1')

# Tieni solo le prime due colonne
spam_dataset = spam_dataset.iloc[:, :2]

spam_dataset.columns = ['label', 'message']
print(spam_dataset.head())

spam_dataset['label'] = spam_dataset['label'].map({'ham': 0, 'spam': 1})
spam_dataset['message'] = spam_dataset['message'].apply(preprocess)


# Controllo
print(spam_dataset.label.value_counts())
print(spam_dataset.isnull().sum())

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(
    spam_dataset['message'], 
    spam_dataset['label'], 
    test_size=0.1, 
    random_state=42,
    stratify=spam_dataset['label']
)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)  
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

#Valuta questi nuovi messaggi:
new_messages = [
                "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now!",
                "Hey, are we still on for lunch tomorrow?",
                "URGENT! Your account has been suspended. Click here to resolve immediately.",
                "Can you send me the report by 5 PM today?",
                "WINNER! Call 555-123-4567 to claim your prize. Limited time only!",
                "Are you free this weekend for the party?",
                "You have been pre-approved for a $5000 loan. Apply now!",
                "Happy birthday! Hope you have a great day!",
                "Claim your free iPhone 14 now! Offer expires soon.",
                "Don't forget about the team meeting at 3 PM." ]

new_messages_processed = [preprocess(msg) for msg in new_messages]
new_messages_tfidf = tfidf.transform(new_messages_processed)
new_predictions = model.predict(new_messages_tfidf)


print("\nNew Messages Predictions:")
for msg, pred in zip(new_messages, new_predictions):
    label = 'spam' if pred == 1 else 'ham'
    print(f'[{label}] Message: "{msg}"')


model_svm = SVC(kernel='linear')
model_svm.fit(X_train_tfidf, y_train)
y_pred_svm = model_svm.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy SVM: {accuracy_svm:.2f}')
print(classification_report(y_test, y_pred_svm, target_names=['ham', 'spam']))

# Valuta i nuovi messaggi con SVM
new_predictions_svm = model_svm.predict(new_messages_tfidf)
print("\nNew Messages Predictions with SVM:")
for msg, pred in zip(new_messages, new_predictions_svm):
    label = 'spam' if pred == 1 else 'ham'
    print(f'[{label}] Message: "{msg}"')


