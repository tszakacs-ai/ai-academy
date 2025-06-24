import re
import os
import ssl
import certifi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Fix per SSL - SOLUZIONI MULTIPLE
def setup_ssl():
    """Configura SSL per evitare errori di certificato"""
    try:
        # Metodo 1: Usa certificati di certifi
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Metodo 2: Configura SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ssl_context
        
        print("✅ SSL configurato correttamente")
        return True
    except Exception as e:
        print(f"Errore configurazione SSL: {e}")
        return False

class TinyLlamaEsercizio:
    """Sistema semplice con TinyLlama per gestire il tuo esercizio"""
    
    def __init__(self, offline_mode=False):
        print("Caricamento TinyLlama...")
        
        # Setup SSL prima di tutto
        setup_ssl()
        
        # Path locale se hai già scaricato il modello
        local_path = "./TinyLlama-1.1B-Chat-v1.0"
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            # Prova prima modalità offline (se esiste già)
            if offline_mode or os.path.exists(local_path):
                print("Caricamento da file locali...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path if os.path.exists(local_path) else model_name,
                    local_files_only=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path if os.path.exists(local_path) else model_name,
                    local_files_only=True
                )
            else:
                # Download con configurazioni SSL
                print("Download da Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_auth_token=False,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_auth_token=False,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # Usa float32 per compatibilità
                )
                
                # Opzionale: salva localmente per usi futuri
                print("Salvataggio locale per usi futuri...")
                self.tokenizer.save_pretrained(local_path)
                self.model.save_pretrained(local_path)
                
        except Exception as e:
            print(f" Errore caricamento: {e}")
            print("\n SOLUZIONI ALTERNATIVE:")
            print("1. Installa certificati: pip install --upgrade certifi requests")
            print("2. Usa modalità offline: TinyLlamaEsercizio(offline_mode=True)")
            print("3. Download manuale da: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            raise
        
        # Configura pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Informazioni aziendali
        self.info_esercizio = """
        ORARI: Lunedì-Sabato 9:00-19:00
        INFORMAZIONI PERSONALI: Mario Rossi ha ricevuto un bonifico sul IBAN IT60X0542811101000000123456.
        Lucia Bianchi ha ricevuto un bonifico sul IBAN IT60X0542811101001000034562.
        Carlo Verdi ha ricevuto un bonifico sul IBAN IT60X0542811101001000067544.
        TELEFONO: 02-1234567
        INDIRIZZO: Via Roma 123, Milano
        """
        
        print("✅ TinyLlama pronto!")
    
    def rileva_dati_sensibili(self, testo):
        """Rileva nomi, telefoni, email nel testo"""
        dati_sensibili = []
        
        # Cerca nomi (2 parole con maiuscola)
        nomi = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', testo)
        dati_sensibili.extend(nomi)
        
        # Cerca telefoni italiani
        telefoni_ita = re.findall(r'\b(?:\+39[-.\s]?)?\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', testo)
        telefoni_usa = re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', testo)
        dati_sensibili.extend(telefoni_ita + telefoni_usa)
        
        # Cerca email
        email = re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', testo)
        dati_sensibili.extend(email)
        
        return dati_sensibili
    
    def anonimizza(self, testo):
        """Sostituisce dati sensibili con placeholder"""
        # Sostituisce nomi
        testo = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NOME]', testo)
        
        # Sostituisce telefoni
        testo = re.sub(r'\b(?:\+39[-.\s]?)?\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', '[TELEFONO]', testo)
        testo = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[TELEFONO]', testo)
        
        # Sostituisce email
        testo = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', testo)
        
        return testo
    
    def genera_risposta(self, domanda):
        """Genera risposta con TinyLlama"""
        
        # Prompt migliorato
        prompt = f"""<|system|>
Sei l'assistente di una consulenza aziendale professionale. Rispondi in modo cortese e preciso.

Informazioni azienda:
{self.info_esercizio}

<|user|>
{domanda}

<|assistant|>
"""
        
        try:
            # Genera con TinyLlama
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=400, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decodifica solo la parte generata
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            risposta = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Pulisci la risposta
            risposta = risposta.strip()
            if risposta.startswith("assistant"):
                risposta = risposta[9:].strip()
            
            return risposta if risposta else "Mi dispiace, posso aiutarti con informazioni sulla nostra azienda."
            
        except Exception as e:
            print(f"⚠️ Errore generazione: {e}")
            return "Mi dispiace, c'è stato un problema tecnico. Come posso aiutarti?"
    
    def processa(self, testo_cliente):
        """Processa richiesta cliente"""
        print(f"📝 Cliente: {testo_cliente}")
        
        # 1. Controlla dati sensibili
        dati_sensibili = self.rileva_dati_sensibili(testo_cliente)
        
        if dati_sensibili:
            print(f"⚠️  DATI SENSIBILI RILEVATI: {dati_sensibili}")
            testo_sicuro = self.anonimizza(testo_cliente)
            print(f"🔒 Testo anonimizzato: {testo_sicuro}")
        else:
            print("✅ Nessun dato sensibile rilevato")
            testo_sicuro = testo_cliente
        
        # 2. Genera risposta
        risposta = self.genera_risposta(testo_sicuro)
        print(f"🤖 Risposta: {risposta}")
        
        # 3. Indica se è sicuro per il cloud
        sicuro_cloud = len(dati_sensibili) == 0
        print(f"☁️  Sicuro per cloud: {'SÌ' if sicuro_cloud else 'NO'}")
        
        return risposta, sicuro_cloud

def installa_dipendenze():
    """Installa dipendenze necessarie"""
    import subprocess
    import sys
    
    packages = ['certifi', 'requests', 'urllib3']
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package])
            print(f"✅ {package} aggiornato")
        except:
            print(f"⚠️ Errore aggiornamento {package}")

def main():
    """Interfaccia semplice con gestione errori"""
    
    print("🔧 Controllo dipendenze...")
    installa_dipendenze()
    
    try:
        # Inizializza sistema
        sistema = TinyLlamaEsercizio()
        
        print("\n" + "="*50)
        print("🏪 SISTEMA ESERCIZIO CON TINYLLAMA")
        print("="*50)
        print("Digita le domande dei clienti (o 'quit' per uscire)")
        
        while True:
            print("\n" + "-"*30)
            domanda = input("Cliente: ").strip()
            
            if domanda.lower() in ['quit', 'q', 'exit']:
                print("👋 Arrivederci!")
                break
            
            if domanda:
                sistema.processa(domanda)
                
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        print("\n🛠️ SOLUZIONI:")
        print("1. Esegui: pip install --upgrade certifi requests urllib3")
        print("2. Controlla la connessione internet")
        print("3. Prova modalità offline se hai già i file")
        print("4. Download manuale del modello da HuggingFace")

if __name__ == "__main__":
    main()