# SMS Sentiment Analysis con Transfer Learning
# pip install pandas scikit-learn transformers torch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

class SMSSentimentAnalyzer:
    def __init__(self):
        self.foundation_model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        
    def load_data(self, file_path):
        """Carica dataset SMS e converte le etichette"""
        # Carica con encoding robusto
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except:
                continue
                
        if df is None:
            raise ValueError("Impossibile caricare il file")
        
        # Auto-rilevamento colonne
        label_col, message_col = self._detect_columns(df)
        
        # Crea dataframe pulito
        df_clean = pd.DataFrame({
            'original_label': df[label_col],
            'message': df[message_col]
        })
        
        # Converti etichette spam/ham in sentiment
        df_clean['sentiment'] = df_clean['original_label'].apply(self._convert_to_sentiment)
        df_clean = df_clean.dropna()
        
        print(f"Dataset caricato: {len(df_clean)} messaggi")
        print(f"Conversione etichette:")
        print(f"  spam -> negativo: {(df_clean['sentiment'] == 'negativo').sum()}")
        print(f"  ham -> positivo/neutro: {(df_clean['sentiment'] != 'negativo').sum()}")
        
        return df_clean
    
    def _detect_columns(self, df):
        """Rileva colonne automaticamente"""
        label_col = None
        message_col = None
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = str(df[col].unique()).lower()
                if 'spam' in unique_vals or 'ham' in unique_vals:
                    label_col = col
                elif message_col is None:
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 20:
                        message_col = col
        
        if not label_col or not message_col:
            cols = list(df.columns)
            label_col = cols[0]
            message_col = cols[1]
            
        return label_col, message_col
    
    def _convert_to_sentiment(self, label):
        """Converte spam/ham in sentiment"""
        label = str(label).lower()
        if 'spam' in label:
            return 'negativo'
        elif 'ham' in label:
            # Per i messaggi ham, usa il foundation model per classificare
            return 'positivo'  # Default, verrà raffinato dal foundation model
        return 'neutro'
    
    def load_foundation_model(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """Carica foundation model per sentiment analysis"""
        print(f"Caricamento foundation model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.foundation_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.foundation_model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            print("Foundation model caricato con successo")
        except Exception as e:
            print(f"Errore nel caricamento del foundation model: {e}")
            print("Usando fallback model...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", return_all_scores=True)
    
    def apply_transfer_learning(self, df):
        """Applica transfer learning per riclassificare i messaggi ham"""
        print("Applicazione transfer learning sui messaggi non-spam...")
        
        # Filtra solo messaggi ham per riclassificazione
        ham_messages = df[df['original_label'].str.lower().str.contains('ham')].copy()
        
        if len(ham_messages) == 0:
            print("Nessun messaggio ham trovato")
            return df
        
        # Applica sentiment analysis ai messaggi ham
        sentiments = []
        for message in ham_messages['message']:
            try:
                result = self.sentiment_pipeline(str(message)[:512])  # Limita lunghezza
                
                # Estrae sentiment dominante
                if isinstance(result[0], list):
                    best_sentiment = max(result[0], key=lambda x: x['score'])
                else:
                    best_sentiment = result[0]
                
                # Mappa le etichette del modello
                label = best_sentiment['label'].lower()
                if 'positive' in label or 'pos' in label:
                    sentiments.append('positivo')
                elif 'negative' in label or 'neg' in label:
                    sentiments.append('negativo')
                else:
                    sentiments.append('neutro')
                    
            except Exception as e:
                print(f"Errore nell'analisi del messaggio: {e}")
                sentiments.append('neutro')
        
        # Aggiorna sentiment per messaggi ham
        df.loc[df['original_label'].str.lower().str.contains('ham'), 'sentiment'] = sentiments
        
        print("Transfer learning completato")
        return df
    
    def evaluate_model(self, df):
        """Valuta le performance del modello"""
        print("\nValutazione del modello:")
        
        # Distribuzione finale
        sentiment_dist = df['sentiment'].value_counts()
        print(f"Distribuzione sentiment finale:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        # Campiona alcuni esempi per verifica manuale
        print(f"\nEsempi di classificazione:")
        for sentiment in df['sentiment'].unique():
            examples = df[df['sentiment'] == sentiment].sample(min(3, len(df[df['sentiment'] == sentiment])))
            print(f"\n{sentiment.upper()}:")
            for _, row in examples.iterrows():
                print(f"  '{row['message'][:80]}...'")
    
    def predict_sentiment(self, message):
        """Predice sentiment di un nuovo messaggio"""
        if self.sentiment_pipeline is None:
            raise ValueError("Foundation model non caricato")
        
        try:
            result = self.sentiment_pipeline(str(message)[:512])
            
            if isinstance(result[0], list):
                scores = {item['label']: item['score'] for item in result[0]}
                best_sentiment = max(result[0], key=lambda x: x['score'])
            else:
                scores = {result[0]['label']: result[0]['score']}
                best_sentiment = result[0]
            
            # Mappa etichette
            label = best_sentiment['label'].lower()
            if 'positive' in label or 'pos' in label:
                sentiment = 'positivo'
            elif 'negative' in label or 'neg' in label:
                sentiment = 'negativo'
            else:
                sentiment = 'neutro'
            
            return {
                'sentiment': sentiment,
                'confidence': best_sentiment['score'],
                'all_scores': scores
            }
            
        except Exception as e:
            print(f"Errore nella predizione: {e}")
            return {'sentiment': 'neutro', 'confidence': 0.0, 'all_scores': {}}

def main():
    print("SMS Sentiment Analysis con Transfer Learning")
    print("=" * 50)
    
    # Inizializza analyzer
    analyzer = SMSSentimentAnalyzer()
    
    # Carica foundation model
    print("\nCaricamento foundation model...")
    analyzer.load_foundation_model()
    
    # Carica dataset
    print("\nCaricamento dataset...")
    file_path = "spam.csv"
    df = analyzer.load_data(file_path)
    
    # Applica transfer learning
    print("\nApplicazione transfer learning...")
    df = analyzer.apply_transfer_learning(df)
    
    # Valuta risultati
    analyzer.evaluate_model(df)
    
    # Test interattivo
    print("\nTest interattivo - Inserisci un messaggio (o 'quit' per uscire):")
    
    while True:
        message = input("\nMessaggio: ")
        if message.lower() == 'quit':
            break
        
        if message.strip():
            result = analyzer.predict_sentiment(message)
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidenza: {result['confidence']:.3f}")
            
            # Mostra tutti i punteggi se disponibili
            if result['all_scores']:
                print("Punteggi dettagliati:")
                for label, score in result['all_scores'].items():
                    print(f"  {label}: {score:.3f}")

if __name__ == "__main__":
    main()