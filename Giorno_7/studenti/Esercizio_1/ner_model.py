from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

class MultilingualNER:
    def __init__(self, model_name="Davlan/bert-base-multilingual-cased-ner-hrl"):
        """
        Inizializza il modello NER multilinguale.
        
        Args:
            model_name (str): Nome del modello Hugging Face
        """
        print(f"Caricamento del modello: {model_name}")
        
        try:
            # Carica tokenizer e modello
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Crea pipeline NER
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Aggrega i token dello stesso tipo
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✅ Modello caricato con successo!")
            if torch.cuda.is_available():
                print("🚀 Utilizzo GPU per l'inferenza")
            else:
                print("💻 Utilizzo CPU per l'inferenza")
                
        except Exception as e:
            print(f"❌ Errore nel caricamento del modello: {e}")
            raise
    
    def extract_entities(self, text, confidence_threshold=0.9):
        """
        Estrae entità nominate dal testo.
        
        Args:
            text (str): Testo da analizzare
            confidence_threshold (float): Soglia di confidenza minima
            
        Returns:
            list: Lista di entità trovate
        """
        try:
            # Esegui NER
            entities = self.ner_pipeline(text)
            
            # Filtra per confidenza
            filtered_entities = [
                entity for entity in entities 
                if entity['score'] >= confidence_threshold
            ]
            
            return filtered_entities
            
        except Exception as e:
            print(f"❌ Errore nell'estrazione: {e}")
            return []
    
    def format_results(self, entities):
        """
        Formatta i risultati in modo leggibile.
        
        Args:
            entities (list): Lista di entità estratte
            
        Returns:
            str: Risultati formattati
        """
        if not entities:
            return "Nessuna entità trovata."
        
        result = "\n🔍 ENTITÀ TROVATE:\n" + "="*50 + "\n"
        
        for i, entity in enumerate(entities, 1):
            result += f"{i}. {entity['word']}\n"
            result += f"   Tipo: {entity['entity_group']}\n"
            result += f"   Confidenza: {entity['score']:.2%}\n"
            result += f"   Posizione: {entity['start']}-{entity['end']}\n\n"
        
        return result
    
    def analyze_text(self, text, confidence_threshold=0.9, show_details=True):
        """
        Analizza un testo e mostra i risultati.
        
        Args:
            text (str): Testo da analizzare
            confidence_threshold (float): Soglia di confidenza
            show_details (bool): Mostra dettagli formattati
        """
        print(f"\n📝 TESTO DA ANALIZZARE:\n{text}\n")
        
        entities = self.extract_entities(text, confidence_threshold)
        
        if show_details:
            print(self.format_results(entities))
        
        return entities

def iban_detector(text):
    """
    Funzione di esempio per rilevare IBAN in un testo.
    
    Args:
        text (str): Testo da analizzare
        
    Returns:
        list: Lista di IBAN trovati
    """
    import re
    iban_pattern = r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'
    return re.findall(iban_pattern, text)