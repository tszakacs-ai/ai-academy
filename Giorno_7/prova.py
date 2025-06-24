import re
import os
import ssl
import certifi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import spacy
from spacy import displacy

# Configurazione SSL
def setup_ssl():
    """Configura SSL per evitare errori di certificato"""
    try:
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ssl_context
        print("✅ SSL configurato")
        return True
    except Exception as e:
        print(f"❌ Errore SSL: {e}")
        return False

class AnonimizzatoreNER:
    """Sistema di anonimizzazione con NER avanzato"""
    
    def __init__(self, offline_mode=False):
        print("🚀 Inizializzazione sistema...")
        
        # Setup SSL
        setup_ssl()
        
        # Carica modello NER italiano/multilingue
        self.carica_ner()
        
        # Carica TinyLlama per le risposte
        self.carica_llm(offline_mode)
        
        # Dati aziendali con informazioni sensibili di test
        self.info_azienda = {
            "orari": "Lunedì-Sabato 9:00-19:00, Domenica chiuso",
            "telefono": "02-1234567",
            "indirizzo": "Via Roma 123, 20121 Milano (MI)",
            "email": "info@azienda.it",
            "clienti": [
                {
                    "nome": "Mario Rossi",
                    "email": "mario.rossi@email.it",
                    "telefono": "339-1234567",
                    "iban": "IT60X0542811101000000123456"
                },
                {
                    "nome": "Lucia Bianchi", 
                    "email": "lucia.bianchi@gmail.com",
                    "telefono": "347-9876543",
                    "iban": "IT90Y0300203280123456789012"
                },
                {
                    "nome": "Carlo Verdi",
                    "email": "c.verdi@outlook.it", 
                    "telefono": "338-5555444",
                    "iban": "IT28W8000000292100645654321"
                },
                {
                    "nome": "Anna Neri",
                    "email": "anna.neri@libero.it",
                    "telefono": "340-1122334",
                    "iban": "IT47L0760103200000012345678"
                }
            ]
        }
        
        print("✅ Sistema pronto!")
    
    def carica_ner(self):
        """Carica modello NER"""
        try:
            # Prova prima con modello italiano
            self.nlp = spacy.load("it_core_news_sm")
            print("✅ NER italiano caricato")
        except OSError:
            try:
                # Fallback su modello inglese
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ NER inglese caricato")
            except OSError:
                print("❌ Nessun modello spaCy trovato!")
                print("Installa con: python -m spacy download it_core_news_sm")
                print("Oppure: python -m spacy download en_core_web_sm")
                raise
    
    def carica_llm(self, offline_mode):
        """Carica TinyLlama"""
        local_path = "./TinyLlama-1.1B-Chat-v1.0"
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            if offline_mode or os.path.exists(local_path):
                print("📂 Caricamento da file locali...")
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True)
            else:
                print("🌐 Download da Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32
                )
                
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ TinyLlama caricato")
            
        except Exception as e:
            print(f"❌ Errore caricamento LLM: {e}")
            raise
    
    def rileva_entita_ner(self, testo):
        """Rileva entità con spaCy NER"""
        doc = self.nlp(testo)
        entita = []
        
        for ent in doc.ents:
            entita.append({
                'testo': ent.text,
                'tipo': ent.label_,
                'inizio': ent.start_char,
                'fine': ent.end_char,
                'descrizione': spacy.explain(ent.label_)
            })
        
        return entita
    
    def rileva_pattern_sensibili(self, testo):
        """Rileva pattern con regex per dati specifici italiani"""
        pattern_sensibili = []
        
        # IBAN italiani
        iban_pattern = r'\bIT\d{2}[A-Z]\d{4}\d{4}\d{12}\b'
        for match in re.finditer(iban_pattern, testo):
            pattern_sensibili.append({
                'testo': match.group(),
                'tipo': 'IBAN',
                'inizio': match.start(),
                'fine': match.end(),
                'descrizione': 'Codice IBAN bancario'
            })
        
        # Codice fiscale
        cf_pattern = r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b'
        for match in re.finditer(cf_pattern, testo):
            pattern_sensibili.append({
                'testo': match.group(),
                'tipo': 'CODICE_FISCALE', 
                'inizio': match.start(),
                'fine': match.end(),
                'descrizione': 'Codice fiscale'
            })
        
        # Telefoni italiani
        tel_pattern = r'\b(?:\+39[-.\s]?)?\d{3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
        for match in re.finditer(tel_pattern, testo):
            pattern_sensibili.append({
                'testo': match.group(),
                'tipo': 'TELEFONO',
                'inizio': match.start(), 
                'fine': match.end(),
                'descrizione': 'Numero di telefono'
            })
        
        # Email
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        for match in re.finditer(email_pattern, testo):
            pattern_sensibili.append({
                'testo': match.group(),
                'tipo': 'EMAIL',
                'inizio': match.start(),
                'fine': match.end(), 
                'descrizione': 'Indirizzo email'
            })
        
        return pattern_sensibili
    
    def anonimizza_testo(self, testo, entita_sensibili):
        """Anonimizza il testo sostituendo le entità sensibili"""
        # Ordina per posizione (dal fondo all'inizio per non alterare gli indici)
        entita_ordinate = sorted(entita_sensibili, key=lambda x: x['inizio'], reverse=True)
        
        testo_anonimo = testo
        sostituzioni = {}
        
        for entita in entita_ordinate:
            placeholder = f"[{entita['tipo']}]"
            testo_anonimo = (
                testo_anonimo[:entita['inizio']] + 
                placeholder + 
                testo_anonimo[entita['fine']:]
            )
            sostituzioni[entita['testo']] = placeholder
        
        return testo_anonimo, sostituzioni
    
    def genera_risposta(self, domanda_anonima):
        """Genera risposta con TinyLlama"""
        
        info_str = f"""
INFORMAZIONI AZIENDA:
- Orari: {self.info_azienda['orari']}
- Telefono: {self.info_azienda['telefono']}
- Indirizzo: {self.info_azienda['indirizzo']}
- Email: {self.info_azienda['email']}
"""
        
        prompt = f"""<|system|>
Sei un assistente di un'azienda italiana. Rispondi in modo cortese e professionale.

{info_str}

<|user|>
{domanda_anonima}

<|assistant|>
"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt", 
                max_length=512,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            risposta = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            return risposta if risposta else "Come posso aiutarla?"
            
        except Exception as e:
            print(f"❌ Errore generazione: {e}")
            return "Mi scusi, c'è stato un problema tecnico. Come posso aiutarla?"
    
    def processa_richiesta(self, testo_cliente):
        """Processa richiesta completa del cliente"""
        print(f"\n📝 CLIENTE: {testo_cliente}")
        print("-" * 60)
        
        # 1. Rileva entità con NER
        entita_ner = self.rileva_entita_ner(testo_cliente)
        
        # 2. Rileva pattern con regex
        pattern_sensibili = self.rileva_pattern_sensibili(testo_cliente)
        
        # 3. Combina tutte le entità sensibili
        tutti_sensibili = []
        
        # Filtra entità NER sensibili (persone, organizzazioni, etc.)
        tipi_sensibili_ner = ['PERSON', 'PER', 'ORG', 'LOC', 'GPE', 'IBAN']
        for ent in entita_ner:
            if ent['tipo'] in tipi_sensibili_ner:
                tutti_sensibili.append(ent)
        
        # Aggiungi pattern regex
        tutti_sensibili.extend(pattern_sensibili)
        
        # 4. Mostra analisi
        if tutti_sensibili:
            print("⚠️  DATI SENSIBILI RILEVATI:")
            for ent in tutti_sensibili:
                print(f"   • {ent['testo']} → {ent['tipo']} ({ent['descrizione']})")
        else:
            print("✅ NESSUN DATO SENSIBILE RILEVATO")
        
        # 5. Anonimizza
        testo_anonimo, sostituzioni = self.anonimizza_testo(testo_cliente, tutti_sensibili)
        
        if sostituzioni:
            print(f"\n🔒 TESTO ANONIMIZZATO: {testo_anonimo}")
        
        # 6. Genera risposta
        risposta = self.genera_risposta(testo_anonimo)
        print(f"\n🤖 RISPOSTA: {risposta}")
        
        # 7. Valuta sicurezza cloud
        sicuro_cloud = len(tutti_sensibili) == 0
        print(f"\n☁️  SICURO PER CLOUD: {'✅ SÌ' if sicuro_cloud else '❌ NO'}")
        
        return {
            'risposta': risposta,
            'sicuro_cloud': sicuro_cloud,
            'dati_sensibili': tutti_sensibili,
            'testo_anonimo': testo_anonimo,
            'sostituzioni': sostituzioni
        }

def main():
    """Interfaccia principale"""
    try:
        print("🔧 Inizializzazione sistema...")
        sistema = AnonimizzatoreNER()
        
        print("\n" + "=" * 60)
        print("🏪 SISTEMA ANONIMIZZAZIONE DATI SENSIBILI")
        print("=" * 60)
        print("Inserisci le richieste dei clienti (o 'quit' per uscire)")
        print("Esempi con dati sensibili:")
        print("- Ciao, sono Mario Rossi, il mio IBAN è IT60X0542811101000000123456")
        print("- Chiamatemi al 339-1234567 o scrivete a mario@email.it")
        
        while True:
            print("\n" + "=" * 30)
            richiesta = input("CLIENTE: ").strip()
            
            if richiesta.lower() in ['quit', 'q', 'exit', 'esci']:
                print("👋 Arrivederci!")
                break
            
            if richiesta:
                sistema.processa_richiesta(richiesta)
                
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        print("\n🛠️ SOLUZIONI:")
        print("1. Installa spaCy: pip install spacy")
        print("2. Scarica modello: python -m spacy download it_core_news_sm")
        print("3. Installa dipendenze: pip install transformers torch certifi")

if __name__ == "__main__":
    main()