import os
import json
from dotenv import load_dotenv
import tkinter as tk
from tkinter import scrolledtext, messagebox

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# === CONFIGURAZIONE API AZURE ===
load_dotenv(dotenv_path=r"C:\Users\BG726XR\OneDrive - EY\Desktop\academy_profice\ai-academy-1\.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_base=os.getenv("AZURE_ENDPOINT_4o"),
    openai_api_version="2024-12-01-preview",
    openai_api_key=os.getenv("AZURE_API_KEY_4o"),
    temperature=0.3
)

# === TEMPLATE PROMPT ===
riepilogo_prompt = PromptTemplate(
    input_variables=["testo"],
    template="Fornisci un riepilogo conciso del seguente testo:\n\n{testo}"
)

analisi_prompt = PromptTemplate(
    input_variables=["testo"],
    template="Fornisci un'analisi semantica dettagliata del seguente testo:\n\n{testo}"
)

risposta_prompt = PromptTemplate(
    input_variables=["testo"],
    template="Prepara una risposta formale al cliente sulla base di questo testo:\n\n{testo}"
)

# === CHAIN PER OGNI FUNZIONE ===
riepilogo_chain = LLMChain(llm=llm, prompt=riepilogo_prompt)
analisi_chain = LLMChain(llm=llm, prompt=analisi_prompt)
risposta_chain = LLMChain(llm=llm, prompt=risposta_prompt)

# === FUNZIONE REINSERIMENTO ENTITÀ ===
def reinserisci_entita(testo, mapping, consentite):
    for tag, valore in mapping.items():
        if any(cons in tag for cons in consentite):
            testo = testo.replace(tag, valore)
        else:
            testo = testo.replace(tag, "[OMISSIS]")
    return testo

# === FUNZIONE DI GESTIONE CHAT ===
def invia_messaggio():
    testo_anonimizzato = input_text.get("1.0", tk.END).strip()

    if not testo_anonimizzato:
        messagebox.showwarning("Attenzione", "Inserisci un testo da analizzare.")
        return

    try:
        with open("mapping.json", "r", encoding="utf-8") as mapfile:
            mapping = json.load(mapfile)
    except FileNotFoundError:
        mapping = {}

    riepilogo = riepilogo_chain.run(testo=testo_anonimizzato)
    analisi = analisi_chain.run(testo=testo_anonimizzato)
    risposta = risposta_chain.run(testo=testo_anonimizzato)

    consentite = ["[PER]_"]  # Aggiornare qui i tag consentiti
    risposta_finale = reinserisci_entita(risposta, mapping, consentite)

    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, f"\n=== Riepilogo ===\n{riepilogo}\n")
    output_text.insert(tk.END, f"\n=== Analisi Semantica ===\n{analisi}\n")
    output_text.insert(tk.END, f"\n=== Risposta Cliente (con reinserimento) ===\n{risposta_finale}\n")
    output_text.insert(tk.END, "\n"+"-"*60+"\n")
    output_text.config(state=tk.DISABLED)
    input_text.delete("1.0", tk.END)



if __name__ == "__main__":
    # === INTERFACCIA GRAFICA ===
    window = tk.Tk()
    window.title("Azure Chatbot LangChain")
    window.geometry("900x600")

    input_label = tk.Label(window, text="Testo anonimizzato:", font=("Arial", 12))
    input_label.pack(pady=5)
    input_text = scrolledtext.ScrolledText(window, height=6, wrap=tk.WORD, font=("Arial", 11))
    input_text.pack(padx=10, fill=tk.BOTH)

    send_button = tk.Button(window, text="Invia", command=invia_messaggio, font=("Arial", 12), bg="#0078D4", fg="white")
    send_button.pack(pady=10)

    output_label = tk.Label(window, text="Output analisi:", font=("Arial", 12))
    output_label.pack(pady=5)
    output_text = scrolledtext.ScrolledText(window, height=20, wrap=tk.WORD, font=("Arial", 11), state=tk.DISABLED)
    output_text.pack(padx=10, fill=tk.BOTH, expand=True)

    window.mainloop()