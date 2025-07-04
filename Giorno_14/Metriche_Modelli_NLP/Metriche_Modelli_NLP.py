# Esempio di valutazione per NER e GPT

from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# --- Esempio NER ---
# Etichette vere e predette (esempio)
true_entities = ['ORG', 'DATE', 'PERSON', 'ORG']
pred_entities = ['ORG', 'DATE', 'ORG', 'ORG']

precision, recall, f1, _ = precision_recall_fscore_support(
    true_entities, pred_entities, average=None, labels=list(set(true_entities))
)

print("NER Metrics:")
for label, p, r, f in zip(set(true_entities), precision, recall, f1):
    print(f"Entity: {label} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f:.2f}")

# --- Esempio GPT (Generazione Testo) ---
# Risposta generata e gold
generated = "La società ACME ha firmato il contratto il 5 luglio."
gold = "Il contratto è stato firmato da ACME il 5 luglio."

# BLEU
bleu = sentence_bleu([gold.split()], generated.split())
print(f"\nBLEU: {bleu:.2f}")

# ROUGE
rouge = Rouge()
scores = rouge.get_scores(generated, gold)
print(f"ROUGE: {scores[0]}")

# (Opzionale) BERTScore richiede installazione e modello, qui solo esempio:
# from bert_score import score
# P, R, F1 = score([generated], [gold], lang="it")
# print(f"BERTScore F1: {F1.mean().item():.2f}")

# --- Come usare ---
# 1. Scegli le metriche in base al task (NER: precision/recall/F1, GPT: BLEU/ROUGE/BERTScore)
# 2. Prepara dati di test (etichette vere e predette per NER, risposte gold e generate per GPT)
# 3. Calcola le metriche come sopra