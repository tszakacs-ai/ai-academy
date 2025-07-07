from transformers import pipeline
from rouge_score import rouge_scorer
import openai
import urllib3
import os
from dotenv import load_dotenv

# Carica variabili d'ambiente dal file .env
load_dotenv(dotenv_path="./Giorno_15/Metriche_NER_GPT_Esercizio/.env")

# Disabilita warning certificati Hugging Face (solo per sviluppo)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurazione Azure OpenAI tramite variabili d'ambiente
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Inizializza pipeline NER Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Esempio di knowledge base (puoi sostituire/estendere)
knowledge_base = [
    {"titolo": "Intelligenza Artificiale", "testo": "L'intelligenza artificiale è la simulazione di processi intelligenti da parte di sistemi informatici."},
    {"titolo": "Reti Neurali", "testo": "Le reti neurali sono modelli computazionali ispirati al funzionamento del cervello umano."},
    {"titolo": "Machine Learning", "testo": "Il machine learning è una branca dell'AI che permette ai sistemi di apprendere dai dati."}
]

def cerca_documento(query):
    # Semplice ricerca: restituisce il documento che contiene la keyword più rilevante
    for doc in knowledge_base:
        if any(word.lower() in doc["testo"].lower() for word in query.split()):
            return doc["testo"]
    return "Nessun documento rilevante trovato nella knowledge base."

def agente_rag(query):
    # Estrai entità dalla domanda
    entities = ner_pipeline(query)
    ent_str = ", ".join([ent['word'] for ent in entities]) if entities else "nessuna entità"
    # Recupera documento rilevante dalla knowledge base
    doc_rilevante = cerca_documento(query)
    # Prompt per GPT
    prompt = (
        f"Domanda: {query}\n"
        f"Entità rilevate: {ent_str}\n"
        f"Contesto (knowledge base): {doc_rilevante}\n"
        "Rispondi in modo chiaro e conciso utilizzando solo le informazioni del contesto."
    )
    # Chiamata a GPT su Azure
    response = openai.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip(), doc_rilevante

def valuta_rouge(risposta, riferimento):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(riferimento, risposta)
    return scores['rougeL'].fmeasure

def main():
    # 1. Generazione risposta
    query = input("Inserisci la domanda: ")
    risposta, riferimento = agente_rag(query)
    print(f"Risposta agente: {risposta}")
    print(f"Testo knowledge base usato come riferimento: {riferimento}")

    # 2. Valutazione automatica rispetto al documento della knowledge base
    rouge_score = valuta_rouge(risposta, riferimento)
    print(f"ROUGE-L F1 rispetto al documento KB: {rouge_score:.2f}")

    # 3. Feedback umano (opzionale)
    voto = input("Dai un voto alla risposta (1-5): ")
    commento = input("Lascia un commento (opzionale): ")

    # 4. Registrazione risultati
    with open("./Giorno_15/Metriche_NER_GPT_Esercizio/valutazioni.csv", "a", encoding="utf-8") as f:
        f.write(f'"{query}","{risposta}","{riferimento}",{rouge_score},{voto},"{commento}"\n')

    print("Valutazione registrata.")

if __name__ == "__main__":
    main()