# Classificatore SMS Spam/Ham - Versione Completa con Metriche
# pip install pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings('ignore')

class SMSSpamClassifier:
    def __init__(self):
        self.pipeline = None
    
    def preprocess_text(self, text):
        """Preprocessa il testo SMS"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, file_path):
        """Carica i dati dal file CSV"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"File caricato con encoding: {encoding}")
                break
            except:
                continue
        
        if df is None:
            raise ValueError("Impossibile caricare il file")
        
        # Identifica automaticamente le colonne
        label_col = message_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                if any(val in str(unique_vals).lower() for val in ['spam', 'ham']):
                    label_col = col
                elif message_col is None and df[col].astype(str).str.len().mean() > 20:
                    message_col = col
        
        if not label_col or not message_col:
            cols = list(df.columns)
            label_col, message_col = cols[0], cols[1]
        
        df_clean = pd.DataFrame({
            'label': df[label_col].astype(str).str.lower(),
            'message': df[message_col]
        }).dropna()
        
        print(f"Dataset: {len(df_clean)} messaggi")
        print(f"Distribuzione:\n{df_clean['label'].value_counts()}")
        return df_clean
    
    def evaluate_detailed_metrics(self, y_test, y_pred):
        """Calcola metriche dettagliate"""
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print("\n=== METRICHE DETTAGLIATE ===")
        for i, cls in enumerate(self.pipeline.classes_):
            print(f"{cls.upper()}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
        
        print(f"\nMEDIE: Precision={precision_avg:.3f}, Recall={recall_avg:.3f}, F1={f1_avg:.3f}")
        return {'precision': precision_avg, 'recall': recall_avg, 'f1': f1_avg}
    
    def compare_models(self, X_train, X_test, y_train, y_test):
        """Confronta diversi modelli"""
        print("\n=== CONFRONTO MODELLI ===")
        
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        models = {
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            
            accuracy = accuracy_score(y_test, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            results[name] = {'accuracy': accuracy, 'f1': f1}
            print(f"{name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 MIGLIOR MODELLO: {best_model[0]} (F1={best_model[1]['f1']:.3f})")
        return results
    
    def train(self, df):
        """Addestra il modello con valutazione completa"""
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        X = df['processed_message']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Addestra Naive Bayes (modello principale)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))),
            ('classifier', MultinomialNB())
        ])
        
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n=== RISULTATI NAIVE BAYES ===")
        print(f"Accuratezza: {accuracy:.3f}")
        print(f"Report:\n{classification_report(y_test, y_pred)}")
        
        # Metriche dettagliate
        detailed_metrics = self.evaluate_detailed_metrics(y_test, y_pred)
        
        # Confronto modelli
        model_comparison = self.compare_models(X_train, X_test, y_train, y_test)
        
        return accuracy, detailed_metrics, model_comparison
    
    def predict(self, message):
        """Predice spam/ham"""
        if self.pipeline is None:
            raise ValueError("Modello non addestrato")
        
        processed = self.preprocess_text(message)
        prediction = self.pipeline.predict([processed])[0]
        probabilities = self.pipeline.predict_proba([processed])[0]
        
        classes = self.pipeline.classes_
        spam_idx = list(classes).index('spam') if 'spam' in classes else 0
        spam_prob = probabilities[spam_idx]
        
        return {
            'prediction': prediction,
            'spam_probability': spam_prob,
            'confidence': max(probabilities)
        }

def main():
    file_path = "spam.csv"
    
    classifier = SMSSpamClassifier()
    
    print("=== CARICAMENTO DATASET ===")
    df = classifier.load_data(file_path)
    
    print("\n=== ADDESTRAMENTO E VALUTAZIONE ===")
    accuracy, detailed_metrics, model_comparison = classifier.train(df)
    
    print("\n=== TEST INTERATTIVO ===")
    print("Scrivi un messaggio per testare (o 'quit' per uscire):")
    
    while True:
        message = input("\n> ")
        if message.lower() == 'quit':
            break
        
        if message.strip():
            result = classifier.predict(message)
            print(f"Predizione: {result['prediction'].upper()}")
            print(f"Spam probability: {result['spam_probability']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()