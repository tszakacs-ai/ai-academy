import os
import certifi
from huggingface_hub import hf_hub_download, HfApi

# Forza HuggingFace Hub a usare il certificato custom (Zscaler incluso)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

from transformers import pipeline
 
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
 
result = classifier("I love using Hugging Face models!")
print(result)