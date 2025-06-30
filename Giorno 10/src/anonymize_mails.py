import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from dotenv import load_dotenv
import openai
import shutil
import logging
from pathlib import Path
import time

import os
import certifi
from huggingface_hub import hf_hub_download, HfApi

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Forza HuggingFace Hub a usare il certificato custom (Zscaler incluso)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

from transformers import pipeline

ner_pipeline = pipeline("ner", 
                   model="Davlan/bert-base-multilingual-cased-ner-hrl",
                   aggregation_strategy="simple"
                   )
# Path to your local model folder
#model_path = r"C:\desktopnodrive\ai-academy\xlm-roberta-base-ner-hrl"
 
# Load tokenizer and model
#tokenizer =AutoTokenizer.from_pretrained(model_path, use_fast=False)
#model = AutoModelForTokenClassification.from_pretrained(model_path)
 
# Create the pipeline
#ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")       
 
# Load environment variables
load_dotenv()
 
# Chat completion model credentials
azure_chat_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_chat_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_chat_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
 
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

def anonymize_text(text):
    """
    Anonymize sensitive information in a single text.
    
    Args:
        text (str): The text to anonymize
        
    Returns:
        str: Anonymized text
    """
    iban_pattern = r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b'
    iban_pattern1 = r'\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{1,4}){1,7}\b'
    fiscal_code_pattern = r'\b([A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z])\b'
    cell_number_pattern = r'\b(?:\+39[\s\.]?)?3[\d\s\.]{8,12}\b' 
    landline_pattern = r'\b0\d{1,3}([\s\.]?\d{2,4}){1,3}\b' # Italian cell number pattern
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
                "entity": f"B-{entity['entity_group']}",
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
    
    # Add IBAN matches as NER-like dicts
    iban_matches1 = []
    for match in re.finditer(iban_pattern1, text):
        iban_matches1.append({
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
    
    # Add Cell Number matches as NER-like dicts
    phone_number_matches = []
    for match in re.finditer(landline_pattern, text):
        phone_number_matches.append({
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
    results = iban_matches + iban_matches1 + fiscal_code_matches + cell_number_matches + phone_number_matches + email_matches + results
    
    # Reconstruct entities
    entities = []
    current = None
    for ent in results:
        if ent["entity"].startswith("B-"):
            if current:
                entities.append(current)
            current = {
                "entity": ent["entity"][2:],
                "tokens": [ent["word"]],
                "start": ent["start"],
                "end": ent["end"]
            }
        elif ent["entity"].startswith("I-") and current:
            current["tokens"].append(ent["word"])
            current["end"] = ent["end"]
    if current:
        entities.append(current)
    
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
    
    return anonymized_text
 
def anonymize_documents(input_dir, output_dir):
    """
    Anonymize all documents in the input directory and save them to output directory.
    
    Args:
        input_dir (str): Path to the directory with documents to anonymize
        output_dir (str): Path to the directory where anonymized documents will be saved
    
    Returns:
        list: List of paths to the anonymized files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    anonymized_files = []
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        
        # Skip directories or already processed files
        if os.path.isdir(input_file_path) or not filename.endswith('.txt'):
            continue
            
        if os.path.exists(output_file_path):
            # File already anonymized
            anonymized_files.append(output_file_path)
            continue
        
        try:
            # Read the content
            with open(input_file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            # Anonymize the content
            anonymized_text = anonymize_text(text)
            
            # Write the anonymized content
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(anonymized_text)
                
            logger.info(f"Anonymized: {filename}")
            anonymized_files.append(output_file_path)
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    return anonymized_files

def get_chat_response(prompt):
    """Get response from the chat model."""
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
 
def classify_document(file_path):
    """
    Classify the document type using AI.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Document type
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    prompt = DOCUMENT_TYPE_PROMPT.format(documento=text)
    response = get_chat_response(prompt)
    return response

def watch_and_anonymize(documents_dir, anonymized_dir, check_interval=60):
    """
    Watch a directory for new documents and anonymize them automatically.
    
    Args:
        documents_dir (str): Directory to watch for new documents
        anonymized_dir (str): Directory to store anonymized documents
        check_interval (int): How often to check for new files (in seconds)
    """
    logger.info(f"Starting automatic anonymization watcher on {documents_dir}")
    processed_files = set()
    
    while True:
        # Get list of files in the documents directory
        try:
            current_files = set(os.path.join(documents_dir, f) for f in os.listdir(documents_dir) 
                              if os.path.isfile(os.path.join(documents_dir, f)) and f.endswith('.txt'))
            
            # Find new files
            new_files = current_files - processed_files
            
            if new_files:
                logger.info(f"Found {len(new_files)} new document(s) to process")
                for file_path in new_files:
                    try:
                        # Get just the filename
                        filename = os.path.basename(file_path)
                        
                        # Read the content
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            
                        # Anonymize the content
                        anonymized_text = anonymize_text(text)
                        
                        # Write the anonymized content
                        output_path = os.path.join(anonymized_dir, filename)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(anonymized_text)
                            
                        logger.info(f"Anonymized: {filename}")
                        processed_files.add(file_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error in watcher: {e}")
            
        time.sleep(check_interval)

def main():
    # Base directory of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define input and output directories
    documents_dir = os.path.join(base_dir, 'data', 'emails', 'documents')
    anonymized_dir = os.path.join(base_dir, 'data', 'emails', 'anonymized')
    
    # Create directories if they don't exist
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(anonymized_dir, exist_ok=True)
    
    # Process existing documents
    logger.info("Processing existing documents...")
    anonymized_files = anonymize_documents(documents_dir, anonymized_dir)
    logger.info(f"Processed {len(anonymized_files)} document(s)")
    
    # Start watching for new documents
    watch_and_anonymize(documents_dir, anonymized_dir)
 
if __name__ == "__main__":
    main()
