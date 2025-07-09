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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # per lemmatizzazione

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)


df_1 = pd.read_csv('Giorno_13\studenti\palmina_angelini\spam.csv', encoding = 'latin-1')
# Tieni solo le prime due colonne
df_1= df_1.iloc[:, :2]
df_1.columns = ['label', 'message']

df_2= pd.read_csv('Giorno_13\studenti\palmina_angelini\message_generated.csv', encoding = 'latin-1')


print(df_1.head())
print(df_2.head())

# Unisci i due dataset
spam_dataset = pd.concat([df_1, df_2], ignore_index=True)

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



# cambia le etichette spam (1) in negativo 

ham_dataset = spam_dataset[spam_dataset['label'] == 0].copy()


print(ham_dataset.head())


model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Crea pipeline per sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer )


label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Applica la sentiment analysis a tutti i messaggi ham
ham_dataset["label"] = ham_dataset["message"].apply(lambda x: label_map[sentiment_pipeline(x)[0]["label"]])
print(ham_dataset["label"].value_counts())





