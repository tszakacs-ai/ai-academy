from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

login(token="incolla qui il token se necessario")  

# STEP 1 – Caricamento del modello pre-addestrato per ottenere gli embedding
modello = SentenceTransformer('all-MiniLM-L6-v2')

# STEP 2 – Elenco di frasi da analizzare semanticamente
frasi = [
    "Marco ha prenotato un volo per Roma.",
    "Marco ha acquistato un biglietto aereo per Roma.",
    "La temperatura oggi è molto bassa.",
    "Il cielo è coperto e potrebbe nevicare.",
]

# STEP 3 – Calcolo degli embedding per ogni frase
embedding_frasi = modello.encode(frasi, convert_to_tensor=True)

# STEP 4 – Calcolo della matrice di similarità coseno tra tutte le frasi
matrice_similarità = util.cos_sim(embedding_frasi, embedding_frasi)

# STEP 5 – Stampa della matrice di similarità
print("📊 Matrice di similarità tra le frasi:\n")
for i in range(len(frasi)):
    for j in range(len(frasi)):
        sim = matrice_similarità[i][j].item()
        print(f"Similarità tra frase {i+1} e frase {j+1}: {sim:.4f}")
    print("-" * 50)

# STEP 6 – Stampa riepilogativa con le frasi confrontate
print("\n📘 Frasi utilizzate:")
for idx, frase in enumerate(frasi, start=1):
    print(f"Frase {idx}: {frase}")
