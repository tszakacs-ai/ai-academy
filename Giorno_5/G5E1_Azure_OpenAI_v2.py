import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from dotenv import load_dotenv
import openai
            
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")

# Create the pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)       

# Load environment variables
load_dotenv()

# Chat completion model credentials
azure_chat_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_chat_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT")

client = openai.AzureOpenAI(
    api_key=azure_chat_api_key,
    azure_endpoint=azure_chat_endpoint,
    api_version=azure_chat_api_version,
    )
        
DOCUMENT_TYPE_PROMPT = """
Ti fornirò un documento aziendale.

Il tuo compito è determinare il tipo di documento, scegliendo tra i seguenti: 
- Mail
- Nota di credito
- Ordine di acquisto
- Contratto
- Altro

Devi basarti solo sul contenuto del documento fornito.

Ora incollerò il contenuto del documento tra tripli apici. Rispondi semplicemente con il tipo, nulla di più.

Documento:
'''
{documento}
'''
"""


def anonymize_documents():
    os.makedirs("anonymized", exist_ok=True)
    for file in os.listdir("documents"):
        with open(os.path.join("documents", file), "r", encoding="utf-8") as f:
            text = f.read()
        
        iban_pattern = r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'
        fiscal_code_pattern = r'\b([A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z])\b'
        cell_number_pattern = r'\b(?:\+39)?3\d{8,9}\b' # Italian cell number pattern (e.g., 3XXYYYYYYY or +393XXXXXXXXX)
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        
        # Get NER results with simple aggregation strategy
        model_entities = ner_pipeline(text)
        
        # Manually find start and end positions for model results
        for entity in model_entities:
            if 'start' not in entity or entity['start'] is None:
                # Find the entity word in the original text
                entity_text = entity['word']
                # Remove leading ▁ if present (XLM-RoBERTa tokenizer specific)
                clean_text = entity_text.replace('▁', ' ').strip()
                
                # Find position in original text
                position = text.find(clean_text)
                if position != -1:
                    entity['start'] = position
                    entity['end'] = position + len(clean_text)
                    # Update word to match the text in the document
                    entity['word'] = clean_text
        
        # Convert model entities to the same format as regex matches
        results = []
        for entity in model_entities:
            if 'start' in entity and entity['start'] is not None:
                results.append({
                    "entity": f"{entity['entity']}",
                    "score": entity['score'],
                    "index": -1,
                    "word": entity['word'],
                    "start": entity['start'],
                    "end": entity['end']
                })
        
        # Add IBAN matches as NER-like dicts
        iban_matches = []
        for match in re.finditer(iban_pattern, text):
            iban_matches.append({
                "entity": "B-IBAN",
                "score": 1.0,
                "index": -1,
                "word": match.group(),
                "start": match.start(),
                "end": match.end()
            })
            
        # Add Fiscal Code matches as NER-like dicts
        fiscal_code_matches = []
        for match in re.finditer(fiscal_code_pattern, text):
            fiscal_code_matches.append({
                "entity": "B-FISCALCODE",
                "score": 1.0,
                "index": -1,
                "word": match.group(),
                "start": match.start(),
                "end": match.end()
            })
            
        # Add Cell Number matches as NER-like dicts
        cell_number_matches = []
        for match in re.finditer(cell_number_pattern, text):
            cell_number_matches.append({
                "entity": "B-CELLNUMBER",
                "score": 1.0,
                "index": -1,
                "word": match.group(),
                "start": match.start(),
                "end": match.end()
            })
            
        # Add Email Address matches as NER-like dicts
        email_matches = []
        for match in re.finditer(email_pattern, text):
            email_matches.append({
                "entity": "B-EMAIL",
                "score": 1.0,
                "index": -1,
                "word": match.group(),
                "start": match.start(),
                "end": match.end()
            })
            
        # Insert regex matches at the beginning
        results = iban_matches + fiscal_code_matches + cell_number_matches + email_matches + results
        
        # Reconstruct entities
        entities = []
        current = None
        
        # Sort results by start position to ensure correct entity reconstruction
        results.sort(key=lambda x: x["start"])
        
        for ent in results:            
            # Start of a new entity
            if ent["entity"].startswith("B-"):
                if current:
                    entities.append(current)
                current = {
                    "entity": ent["entity"][2:],
                    "tokens": [ent["word"]],
                    "start": ent["start"],
                    "end": ent["end"]
                }
            # Continuation of current entity
            elif ent["entity"].startswith("I-") and current:
                current["tokens"].append(ent["word"])
                current["end"] = ent["end"]
            # Start of a new entity (if no B- prefix was found but it's a different type)
            else:
                if current:
                    entities.append(current)
                current = {
                    "entity": ent["entity"],
                    "tokens": [ent["word"]],
                    "start": ent["start"],
                    "end": ent["end"]
                }
        
        if current:
            entities.append(current)
            
        # Merge adjacent entities of the same type that may have been split incorrectly
        merged_entities = []
        i = 0
        while i < len(entities):
            current = entities[i]
            # Look ahead to see if next entity should be merged
            while i + 1 < len(entities) and entities[i+1]["entity"] == current["entity"] and entities[i+1]["start"] - current["end"] <= 1:
                # Merge with next entity
                current["tokens"].extend(entities[i+1]["tokens"])
                current["end"] = entities[i+1]["end"]
                i += 1
            merged_entities.append(current)
            i += 1
            
        # Map for anonymization
        label_map = {
            "EMAIL": "Email",
            "PER": "Nome",
            "ORG": "Azienda",
            "LOC": "Luogo",
            "IBAN": "IBAN",
            "FISCALCODE": "FISCALCODE",
            "CELLNUMBER": "Cellulare",
        }
        
        # Replace the entities list with our merged list
        entities = merged_entities
   
        entity_counters = {}
        replacements = []
        for ent in entities:
            label = ent["entity"]
            mapped = label_map.get(label, label)
            entity_counters[mapped] = entity_counters.get(mapped, 0) + 1

            # Custom anonymization for IBAN and FISCALCODE
            entity_text = " ".join(ent["tokens"]).strip()
            if label == "IBAN":
                # Mask all but first 4 chars
                masked = entity_text[:4] + "*" * (len(entity_text) - 4)
                placeholder = f"[{masked}]"
            elif label == "FISCALCODE":
                # Mask all but first 3 chars
                masked = entity_text[:3] + "*" * (len(entity_text) - 3) 
                placeholder = f"[{masked}]"
            elif label == "CELLNUMBER":
                masked = entity_text[:3] + "*" * (len(entity_text) - 3)
                placeholder = f"[{masked}]"
            else:
                placeholder = f"[{mapped}_{entity_counters[mapped]}]"

            replacements.append((entity_text, placeholder, ent))
            
        # Sort replacements by start position in reverse order (to avoid position shifts)
        replacements.sort(key=lambda x: x[2]["start"], reverse=True)
        
        # Replace entities in text
        anonymized_chars = list(text)
        
        for entity_text, placeholder, entity in replacements:
            start = entity["start"]
            end = entity["end"]
            
            # Replace the characters at the specified positions
            anonymized_chars[start:end] = placeholder
        
        anonymized_text = ''.join(anonymized_chars)
        # Write anonymized file
        with open(os.path.join("anonymized", file), "w", encoding="utf-8") as f:
            f.write(anonymized_text)

def get_chat_response(prompt):

    response = client.chat.completions.create(
    model=azure_chat_deployment,
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=256,
    temperature=1,
    )
    return response.choices[0].message.content.strip()

def test_documents():
    anonymize_documents()
    for file in os.listdir("anonymized"):
        with open(os.path.join("anonymized", file), "r", encoding="utf-8") as f:
            text = f.read()
        prompt = DOCUMENT_TYPE_PROMPT.format(documento=text)
        response = get_chat_response(prompt)
        print(f"[{file}] → Tipo rilevato: {response}")

if __name__ == "__main__":
    test_documents()



    