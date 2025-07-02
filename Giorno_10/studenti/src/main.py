from modules.ner_anonymizer import NERAnonymizer
from modules.data_loader import load_data
from pathlib import Path
import os, sys
from config import *
from modules.AzureRag import AzureRAGSystem
from dotenv import load_dotenv
from subprocess import run

# === FILE AND MODEL PATHS ===
model_path = MODEL_PATH
folder_path = FOLDER_PATH
anon_file_path = ANON_FILE_PATH
results_path = RESULTS_PATH
history_path = HISTORY_PATH

load_dotenv()
AZURE_ENDPOINT_RAG = os.getenv("AZURE_ENDPOINT_RAG")  
API_KEY_RAG = os.getenv("API_KEY_RAG")
EMBEDDING_DEPLOYMENT_RAG = os.getenv("EMBEDDING_DEPLOYMENT_RAG")  
GPT_DEPLOYMENT_RAG = os.getenv("GPT_DEPLOYMENT_RAG")

                
if __name__ == "__main__":
    print(AZURE_ENDPOINT_RAG, API_KEY_RAG, EMBEDDING_DEPLOYMENT_RAG, GPT_DEPLOYMENT_RAG)
    print("\n\n")

    # === ANONYMIZATION ===
    try:
        print("[INFO] Starting anonymization...")

        # Initialize the anonymizer with the specified model
        anonymizer = NERAnonymizer(str(model_path))

        if os.path.exists(folder_path):
            # Get all files in the folder
            all_items = os.listdir(folder_path)
            # Filter out directories and keep only files
            files_path = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
            # Read each file and append its content to the list
            for path in files_path:
                file_name = path.split("\\")[-1]
                # Anonymize and save to file
                anonymizer.anonymize_txt_file(path, str(anon_file_path) + "\\" + file_name)
                print(f"[SUCCESS] Anonymized file created at: {anon_file_path}")

    except Exception as e:
        print(f"[ERROR] During initialization or anonymization: {e}")
        exit(1)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'streamlit':
            cmd = ['streamlit', 'run', 'streamlit_app.py']
            result = run(
            cmd,
            check=True,  
            text=True    
        )
        
        if sys.argv[1] == 'terminal':
            # === ANALYSIS WITH RAG ===
            # Inizializza il sistema RAG
            rag = AzureRAGSystem(
                azure_endpoint=AZURE_ENDPOINT_RAG,
                api_key=API_KEY_RAG,
                embedding_deployment=EMBEDDING_DEPLOYMENT_RAG,
                gpt_deployment=GPT_DEPLOYMENT_RAG
            )
            print("[INFO] Starting RAG...")
            
            # Carica i documenti anonimizzati
            documents = load_data(anon_file_path)
            print(f"[INFO] Loaded {len(documents)} documents from {anon_file_path}")

            # Aggiungi documenti al sistema
            rag.add_documents(documents)
            print("[INFO] Documents added to RAG system.")
            
            # Inizio conversazione utente
            while True:
                user_input = input("\nquit --> chiudi chat" \
                                "\nclear --> cancella storia" \
                                "\nsummary --> riassunto della storia" \
                                "\nsave --> salva storia in file" \
                                    "\n\nScrivi la tua domanda: ")
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    if command == "quit":
                        print("Chiusura della chat...")
                        break
                    elif command == "clear":
                        rag.clear_chat_history()
                        print("Storia cancellata.")
                    elif command == "summary":
                        summary = rag.get_chat_summary()
                        print(f"Riassunto della storia:\n{summary}")
                    elif command == "save":
                        rag.save_chat_history(str(history_path))
                        print(f"Storia salvata in {history_path}")
                
                response = rag.generate_response(user_input)
                print(f"\nRisposta del sistema:\n{response['response']}")
                print(f"Fonti utilizzate: {response['sources']}")
