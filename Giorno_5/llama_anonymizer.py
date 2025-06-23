import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

class TinyLlamaCSVAnonymizer:
    def __init__(self, model_path=None):
        """Inizializza l'anonimizzatore con TinyLlama"""
        print(f"📥 Caricando TinyLlama da: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set pad token se non esiste
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ TinyLlama caricato con successo!")
        
        # Lista per salvare le righe anonimizzate
        self.anonymized_lines = []
    
    def anonymize_names_in_line(self, line):
        """Usa TinyLlama per sostituire nomi e cognomi in una singola riga"""
        
        prompt = f"""
Original text: "{line.strip()}"
        
You are a data anonymization expert. Replace all first names and last names in the given text with the strings <NOME> and <COGNOME>, but keep everything else exactly the same.

Original text: Mario,Rossi,IT60 X054 2811 1010 0000 0123 456,Apple Inc,€1250.50
Required output: <NOME>,<COGNOME>,IT60 X054 2811 1010 0000 0123 456,Apple Inc,€1250.50

"""
        
        try:
            # Tokenizza il prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Genera la risposta
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decodifica la risposta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Estrai solo la parte anonimizzata
            if "<|assistant|>" in response:
                anonymized_line = response.split("<|assistant|>")[-1].strip()
                
                # Pulisci la risposta da eventuali caratteri extra
                anonymized_line = anonymized_line.replace('"', '').strip()
                
                if anonymized_line and len(anonymized_line) > 0:
                    return anonymized_line
            
        except Exception as e:
            print(f"⚠️  Errore con TinyLlama per la riga: {e}")
        
        # Se il modello fallisce, ritorna la riga originale
        return line.strip()
    
    def process_file(self, input_file="./sample_data.csv"):
        """Legge il file riga per riga e anonimizza i nomi"""
        print(f"\n📄 Leggendo file: {input_file}")
        
        # Verifica che il file esista
        if not Path(input_file).exists():
            print(f"❌ File non trovato: {input_file}")
            return
        
        # Leggi tutte le righe del file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📊 Trovate {len(lines)} righe da processare")
        
        # Processa ogni riga
        for i, line in enumerate(lines):
            # Salta righe vuote
            if line.strip() == "":
                self.anonymized_lines.append("")
                continue
            
            print(f"🔄 Processando riga {i+1}/{len(lines)}: {line.strip()[:50]}...")
            
            # Anonimizza la riga con TinyLlama
            anonymized_line = self.anonymize_names_in_line(line)
            
            # Aggiungi alla lista
            self.anonymized_lines.append(anonymized_line)
            
            print(f"✅ Risultato: {anonymized_line[:50]}...")
        
        print(f"\n🎉 Processamento completato!")
        print(f"📋 Righe originali: {len(lines)}")
        print(f"📋 Righe anonimizzate: {len(self.anonymized_lines)}")
        
        return self.anonymized_lines
    
    def save_anonymized_file(self, output_file="anonymized_output.txt"):
        """Salva la lista delle righe anonimizzate in un file"""
        if not self.anonymized_lines:
            print("❌ Nessuna riga anonimizzata da salvare!")
            return
        
        print(f"💾 Salvando file anonimizzato: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in self.anonymized_lines:
                f.write(line + '\n')
        
        print(f"✅ File salvato con successo!")
        
        # Mostra anteprima
        print(f"\n👀 Anteprima prime 3 righe anonimizzate:")
        for i, line in enumerate(self.anonymized_lines[:3]):
            if line.strip():
                print(f"{i+1}: {line}")

# Funzione di test
def main():
    """Funzione per testare il metodo step by step"""
    print("🦙 Anonimizzazione righe con TinyLlama")
    print("=" * 60)
    
    local_dir = Path(__file__).parent
    data = local_dir / "sample_data.csv"
    model = local_dir / "TinyLlama-1.1B-Chat-v1.0"
    output_file = local_dir / "cleaned_output.txt"

    print(f"📂 Percorso file di input: {data}")

    anonymizer = TinyLlamaCSVAnonymizer(model_path=model)
    anonymized_lines = anonymizer.process_file(str(data))
    anonymizer.save_anonymized_file(str(output_file))

if __name__ == "__main__":
    main()