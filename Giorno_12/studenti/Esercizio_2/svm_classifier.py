import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns


class SpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Preprocessa il testo per migliorare le prestazioni del modello"""
        # Converti in minuscolo
        text = text.lower()
        
        # Rimuovi caratteri speciali e numeri
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenizza
        words = text.split()
        
        # Rimuovi stopwords e applica stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def train(self, X_train, y_train):
        """Addestra il modello SVM"""
        # Preprocessa i testi
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Crea pipeline con TF-IDF e SVM
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,      # Limita il numero di features
                ngram_range=(1, 2),     # Usa unigram e bigram
                min_df=2,               # Ignora termini che appaiono in meno di 2 documenti
                max_df=0.95,            # Ignora termini che appaiono in più del 95% dei documenti
                stop_words='english'
            )),
            ('svm', SVC(
                kernel='rbf',           # Kernel RBF per catturare relazioni non lineari
                C=1.0,                  # Parametro di regolarizzazione
                gamma='scale',          # Parametro del kernel
                probability=True,       # Abilita calcolo probabilità
                random_state=42
            ))
        ])
        
        # Addestra il modello
        print("Addestramento del modello SVM in corso...")
        self.model.fit(X_train_processed, y_train)
        print("Addestramento completato!")
        
    def predict(self, X_test):
        """Fa predizioni sui dati di test"""
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        return self.model.predict(X_test_processed)
    
    def predict_proba(self, X_test):
        """Restituisce le probabilità delle predizioni"""
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        return self.model.predict_proba(X_test_processed)
    
    def evaluate(self, X_test, y_test):
        """Valuta le prestazioni del modello"""
        predictions = self.predict(X_test)
        
        # Calcola metriche
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuratezza: {accuracy:.4f}")
        print("\nReport di Classificazione:")
        print(classification_report(y_test, predictions, 
                                    target_names=['Ham', 'Spam']))
        
        # Matrice di confusione
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.title('Matrice di Confusione')
        plt.ylabel('Valore Reale')
        plt.xlabel('Predizione')
        plt.show()
        
        return accuracy, predictions

# Funzione principale per addestrare il modello
def train_spam_classifier(cleaned_df):
    """
    Addestra un classificatore SVM per spam detection
    
    Args:
        cleaned_df: DataFrame con colonne 'text' e 'label'
    
    Returns:
        classifier: Modello addestrato
        X_test, y_test: Dati di test per ulteriori valutazioni
    """
    
    # Verifica la struttura del dataset
    print("Struttura del dataset:")
    print(f"Dimensioni: {cleaned_df.shape}")
    print(f"Distribuzione delle classi:")
    print(cleaned_df['label'].value_counts())
    print(f"Percentuale spam: {(cleaned_df['label'].sum() / len(cleaned_df)) * 100:.2f}%")
    
    # Dividi i dati in training e test
    X = cleaned_df['text'].values
    y = cleaned_df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDimensioni training set: {len(X_train)}")
    print(f"Dimensioni test set: {len(X_test)}")
    
    # Crea e addestra il classificatore
    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    
    # Valuta il modello
    print("\n" + "="*50)
    print("VALUTAZIONE DEL MODELLO")
    print("="*50)
    
    accuracy, predictions = classifier.evaluate(X_test, y_test)
    
    # Esempi di predizioni
    print("\nEsempi di predizioni:")
    print("-" * 30)
    
    # Prendi alcuni esempi casuali
    indices = np.random.choice(len(X_test), 5, replace=False)
    probabilities = classifier.predict_proba(X_test[indices])
    
    for i, idx in enumerate(indices):
        text_preview = X_test[idx][:100] + "..." if len(X_test[idx]) > 100 else X_test[idx]
        actual = "Spam" if y_test[idx] == 1 else "Ham"
        predicted = "Spam" if predictions[idx] == 1 else "Ham"
        confidence = max(probabilities[i]) * 100
        
        print(f"Testo: {text_preview}")
        print(f"Reale: {actual} | Predetto: {predicted} | Confidenza: {confidence:.1f}%")
        print("-" * 30)
    
    return classifier, X_test, y_test

# Funzione per testare nuovi messaggi
def classify_new_messages(classifier, messages):
    """
    Classifica nuovi messaggi usando il modello addestrato
    
    Args:
        classifier: Modello addestrato
        messages: Lista di stringhe da classificare
    
    Returns:
        results: Lista di tuple (messaggio, predizione, probabilità)
    """
    predictions = classifier.predict(messages)
    probabilities = classifier.predict_proba(messages)
    
    results = []
    for i, message in enumerate(messages):
        pred_label = "Spam" if predictions[i] == 1 else "Ham"
        confidence = max(probabilities[i]) * 100
        results.append((message, pred_label, confidence))
    
    return results
