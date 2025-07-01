import os
from dotenv import load_dotenv
from transformers import pipeline
import re
from crewai import Agent, Task, Crew
from crewai.llm import LLM

# Load .env variables
load_dotenv()
language_key = os.environ.get('LANGUAGE_KEY')
language_endpoint = os.environ.get('LANGUAGE_ENDPOINT')

# Load multilingual NER pipeline
ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

# llm = LLM(
#     model="azure/gpt-4.1",
#     api_version="2025-01-01-preview",
#     api_key=language_key,
#     api_base=language_endpoint,
# )

# -------------------------------
# Helper anonymization function
# -------------------------------
def anonymize_names(text):
    return "[NOME] ha ricevuto un bonifico sull'IBAN [IBAN]."

def anonymize_iban(text):
    iban_pattern = r"\bIT\d{2}[A-Z0-9]{1,23}\b"
    return re.sub(iban_pattern, "[IBAN]", text, flags=re.IGNORECASE)


# -------------------------------
# Agent for CrewAI
# -------------------------------
class AnonymizerAgent(Agent):
    def process(self, text):
        testo_anon = anonymize_names(text)
        testo_anon = anonymize_iban(testo_anon)
        return testo_anon

# -------------------------------
# Agent and CrewAI setup
# -------------------------------
agent = AnonymizerAgent(
    name="Anonimizzatore",
    role="privacy-bot",
    goal="Anonimizzadati sensibili",
    backstory="Un assistente AI incaricato di progettare la privacy dei dati nei documenti aziendali.",
    llm=None,
    openai_api_key=language_key
)

# Testo d'esempio
text = "Mario Rossi ha ricevuto un bonifico sull'IBAN IT60X054281101000000123456."


# -------------------------------
# Task CrewAI setup
# -------------------------------
task = Task(
    description="Anonimizza nomi e IBAN nel testo fornito.",
    expected_output="Il testo deve avere tutti i nomi sostituiti da [NOMI] e gli IBAN da [IBAN].",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task]
)

result = crew.run()
print("Frase anonimizzata:\n", result)
