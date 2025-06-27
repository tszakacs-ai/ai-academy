from ner_anonymizer import NERAnonymizer
from chatbot import ChatbotAnalyzer
from pathlib import Path
import os
from config import *

# === FILE AND MODEL PATHS ===
anon_file_path = ANON_FILE_PATH
folder_path = FOLDER_PATH
model_path = MODEL_PATH
results_path = RESULTS_PATH
parent_dir = Path(folder_path).parent.parent
                
if __name__ == "__main__":

    # === ANONYMIZATION ===
    try:
        print("[INFO] Starting anonymization...")

        # Initialize the anonymizer with the specified model
        anonymizer = NERAnonymizer(model_path)

        if os.path.exists(folder_path):
            # Get all files in the folder
            all_items = os.listdir(folder_path)
            # Filter out directories and keep only files
            files_path = [os.path.join(folder_path, item) for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
            # Read each file and append its content to the list
            for path in files_path:
                file_name = path.split("\\")[-1]
                # Anonymize and save to file
                anonymizer.anonymize_txt_file(path, anon_file_path + "\\" + file_name)
                print(f"[SUCCESS] Anonymized file created at: {anon_file_path}")

    except Exception as e:
        print(f"[ERROR] During initialization or anonymization: {e}")
        exit(1)

    # === ANALYSIS WITH CHATBOT ===
    try:
        print("[INFO] Initializing ChatbotAnalyzer...")

        dotenv_path = parent_dir / '.env'
        
        # Instantiate the analyzer with the combined prompt
        analyzer = ChatbotAnalyzer(
            dotenv_path=dotenv_path,
            use_combined_prompt=True  # Set to False if you prefer separate prompts
        )

        if os.path.exists(anon_file_path):
            # Get all files in the folder
            all_items = os.listdir(anon_file_path)
            # Filter out directories and keep only files
            files_path = [os.path.join(anon_file_path, item) for item in all_items if os.path.isfile(os.path.join(anon_file_path, item))]
            # Read each file and append its content to the list
            for path in files_path:
                # Read the anonymized text
                with open(path, 'r', encoding="utf-8") as f:
                    anon_text = f.read()

                # Run the analysis via LLM
                results = analyzer.process_text(anon_text)
                
                #check if file exists in results_path, if not create it
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                    
                #save results to a file
                with open(path.replace(anon_file_path, results_path), 'a', encoding="utf-8") as f:
                    if "full_output" in results:
                        f.write(results["full_output"])
                    else:
                        f.write("\n=== Summary ===\n" + results.get("summary", "[No summary]") + "\n")
                        f.write("\n=== Semantic Analysis ===\n" + results.get("analysis", "[No analysis]") + "\n")
                        f.write("\n=== Customer Response (with reinsertion) ===\n" + results.get("response", "[No response]") + "\n")

        # print("[INFO] Analysis completed successfully.")
        # # === PRINT RESULTS ===
        # print("\n=== RESULTS ===\n")
        # if "full_output" in results:
        #     print(results["full_output"])
        # else:
        #     print("\n=== Summary ===\n", results.get("summary", "[No summary]"))
        #     print("\n=== Semantic Analysis ===\n", results.get("analysis", "[No analysis]"))
        #     print("\n=== Customer Response (with reinsertion) ===\n", results.get("response", "[No response]"))

    except Exception as e:
        print(f"[ERROR] During initialization or analysis execution: {e}")
