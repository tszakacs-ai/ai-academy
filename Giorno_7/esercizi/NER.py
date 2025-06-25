
import os
from dotenv import load_dotenv

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


load_dotenv()  

# This example requires environment variables named "LANGUAGE_KEY" and "LANGUAGE_ENDPOINT"
language_key = os.environ.get('LANGUAGE_KEY')
language_endpoint = os.environ.get('LANGUAGE_ENDPOINT')


# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()


# Funzione per anonimizzare un testo in base alle entità trovate
def anonymize_text(text, entities):
    for entity in sorted(entities, key=lambda e: e.offset, reverse=True):
        placeholder = f"[{entity.category.upper()}]"
        start = entity.offset
        end = start + entity.length
        text = text[:start] + placeholder + text[end:]
    return text


def entity_recognition_and_anonymization(client, documents):
    try:
        results = client.recognize_entities(documents=documents)

        for i, result in enumerate(results):
            print(f"\nOriginal text:\n{documents[i]}")
            if result.is_error:
                print(f"Error: {result.error}")
                continue

            print("\nNamed Entities:")
            for entity in result.entities:
                print(f"  - {entity.text} ({entity.category}) [{entity.subcategory}] — score: {round(entity.confidence_score, 2)}")

            anonymized = anonymize_text(documents[i], result.entities)
            print("\nAnonymized text:\n" + anonymized)

    except Exception as err:
        print(f"Exception occurred: {err}")

test_texts = [
    "Mario Rossi lives in Rome and his phone number is 345-678-9012.",
    "Laura Bianchi works at Contoso Ltd. Her email is laura.bianchi@email.com.",
    "Giovanni went to Milan last Friday and spent 200 euros.",
    "You can reach Marco at marco@azienda.it or call him at +39 333 1122334.",
    "Serena lives at 1234 Via Napoli, Florence, Italy."
]

entity_recognition_and_anonymization(client, test_texts)