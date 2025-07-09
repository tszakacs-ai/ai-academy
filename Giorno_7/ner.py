import os
import certifi
from huggingface_hub import hf_hub_download, HfApi

# Forza HuggingFace Hub a usare il certificato custom (Zscaler incluso)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

from transformers import pipeline

ner_pipe = pipeline("ner", 
                   model="Davlan/bert-base-multilingual-cased-ner-hrl",
                   aggregation_strategy="simple"
                   )

text = "Mario Rossi è un ingegnere che lavora a Milano. Il suo indirizzo email"
entities = ner_pipe(text)

for entity in entities:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")

