from AzureRag import AzureRAGSystem
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    # Configurazione - sostituisci con i tuoi valori
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")  
    API_KEY = os.getenv("API_KEY")
    EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")  
    GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT")  
    
    # Inizializza il sistema RAG
    rag = AzureRAGSystem(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        embedding_deployment=EMBEDDING_DEPLOYMENT,
        gpt_deployment=GPT_DEPLOYMENT
    )
    
    # Esempio di documenti
    documents = [
        {
            'title': 'Pollaio',
            'content': """Allevo polli."""
        },
        {
            'title': 'Porcilaia',
            'content': """Allevo maiali."""
        },
        {
            'title': 'Pacolo',
            'content': """Allevo pecore."""
        }
    ]
    
    # Aggiungi documenti al sistema
    rag.add_documents(documents)
    
    # Esempio di query
    query = "Dove si allevano polli?"
    
    # Genera risposta
    result = rag.generate_response(query, top_k=2)
    
    print(f"\nDomanda: {query}")
    print(f"\nRisposta: {result['response']}")
    print(f"\nFonti utilizzate:")
    for source in result['sources']:
        print(f"  - {source['title']} (score: {source['score']:.3f})")
    print(f"\nToken utilizzati: {result['tokens_used']}")
    
    # Salva l'indice per uso futuro
    rag.save_index('rag_index')
    
    # Per caricare un indice esistente:
    # rag.load_index('rag_index')