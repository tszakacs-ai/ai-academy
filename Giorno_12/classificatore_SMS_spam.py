# Classificatore SMS Spam/Ham - Versione Compatta e Funzionante
# pip install pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        # Prova diversi encoding
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
            raise ValueError("Impossibile caricare il file con nessun encoding testato")
        
        # Identifica automaticamente le colonne corrette
        print(f"Colonne originali: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
        # Cerca le colonne che contengono label e message
        label_col = None
        message_col = None
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                # Se ha valori come spam/ham è probabilmente la label
                if any(val in str(unique_vals).lower() for val in ['spam', 'ham']):
                    label_col = col
                # La colonna con testi più lunghi è probabilmente il messaggio
                elif message_col is None:
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 20:  # Assumiamo che i messaggi siano più lunghi di 20 caratteri
                        message_col = col
        
        # Se non trova automaticamente, usa le prime due colonne
        if label_col is None or message_col is None:
            cols = list(df.columns)
            label_col = cols[0]
            message_col = cols[1]
        
        # Crea il dataframe pulito
        df_clean = pd.DataFrame({
            'label': df[label_col],
            'message': df[message_col]
        })
        
        # Rimuovi righe vuote
        df_clean = df_clean.dropna()
        
        # Standardizza le label
        df_clean['label'] = df_clean['label'].astype(str).str.lower()
        
        print(f"Dataset caricato: {len(df_clean)} messaggi")
        print(f"Distribuzione classi:\n{df_clean['label'].value_counts()}")
        print(f"Prime 3 righe:\n{df_clean.head(3)}")
        
        return df_clean
    
    def train(self, df):
        """Addestra il modello"""
        # Preprocessa i messaggi
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Separa features e target
        X = df['processed_message']
        y = df['label']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Crea pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))),
            ('classifier', MultinomialNB())
        ])
        
        # Addestra
        self.pipeline.fit(X_train, y_train)
        
        # Valuta
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n=== RISULTATI ===")
        print(f"Accuratezza: {accuracy:.3f}")
        print(f"Report:\n{classification_report(y_test, y_pred)}")
        
        return accuracy
    
    def predict(self, message):
        """Predice spam/ham"""
        if self.pipeline is None:
            raise ValueError("Modello non addestrato")
        
        processed = self.preprocess_text(message)
        prediction = self.pipeline.predict([processed])[0]
        probabilities = self.pipeline.predict_proba([processed])[0]
        
        # Trova la probabilità di spam
        classes = self.pipeline.classes_
        spam_idx = list(classes).index('spam') if 'spam' in classes else 0
        spam_prob = probabilities[spam_idx]
        
        return {
            'prediction': prediction,
            'spam_probability': spam_prob,
            'confidence': max(probabilities)
        }

def main():
    # Il file è nella stessa directory
    file_path = "spam.csv"
    
    classifier = SMSSpamClassifier()
    
    print("=== CARICAMENTO DATASET ===")
    df = classifier.load_data(file_path)
    
    print("\n=== ADDESTRAMENTO ===")
    accuracy = classifier.train(df)
    
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

def analyze_compliance():
    print("\n" + "="*50)
    print("ANALISI EU AI ACT")
    print("="*50)
    print("""
CLASSIFICAZIONE: Sistema a Rischio Limitato
REQUISITI:
• Trasparenza verso gli utenti
• Supervisione umana raccomandata
• Monitoraggio delle prestazioni
• Documentazione del sistema
• Controllo bias e discriminazione

RACCOMANDAZIONI:
1. Informare utenti sull'uso dell'AI
2. Implementare feedback umano
3. Monitorare accuratezza nel tempo
4. Documentare training e dati
5. Test periodici su bias
    """)

if __name__ == "__main__":
    main()
    analyze_compliance()