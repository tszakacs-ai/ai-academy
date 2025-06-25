import openai

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="LA_TUA_API_KEY",  # <-- La tua chiave API di Azure OpenAI
    azure_endpoint="IL_TUO_API_ENDPOINT", # <-- Il tuo  API endpoint di Azure OpenAI
    api_version="2024-12-01-preview",  # <-- La versione dal portale Azure
)

document_text = """MINISTERO DELL'INTERNO
Corpo di Polizia Locale

VERBALE DI ACCERTAMENTO DI VIOLAZIONE AMMINISTRATIVA
Ex art. 201 Codice della Strada

N. Verbale: PL2025/123456
Data violazione: 20/06/2025
Ora violazione: 10:30
Luogo violazione: Via Roma, 12, Milano

IDENTIFICAZIONE VEICOLO
Targa: AB123CD
Marca/Modello: Fiat Panda
Proprietario: Mario Rossi (Nato a Roma il 01/01/1980)

DESCRIZIONE DELLA VIOLAZIONE
Il veicolo sopra identificato è stato accertato in sosta vietata (Art. 158 comma 2 lett. c CdS - sosta in corrispondenza o in prossimità di intersezione) in area segnalata con apposito cartello e segnaletica orizzontale. La violazione è stata rilevata dall'agente [Nome Agente] con matricola [Matricola Agente].

SANZIONE AMMINISTRATIVA
Importo: Euro 42,00
Importo ridotto (entro 5 giorni): Euro 29,40
Termine per il pagamento: 60 giorni dalla data di notifica.
Modalità di pagamento: Bollettino postale allegato o online sul portale del Comune di Milano.

INFORMAZIONI PER IL RICORSO
È possibile presentare ricorso al Prefetto di Milano entro 60 giorni dalla notifica del presente verbale, oppure al Giudice di Pace entro 30 giorni. Per maggiori informazioni consultare il retro del presente verbale.

NOTA BENE: Il presente verbale ha valore di notifica.

Firmato
L'Agente Accertatore
[Firma dell'Agente]
"""


response = client.chat.completions.create(
    model="gtp-4.1",  # <-- Il nome esatto del deployment in Azure
    messages=[
        {"role": "system", "content": "Sei un analizzatore di documenti esperto, ti verranno forniti dei documenti e tu devi identificarne il tipo"},
        {"role": "user", "content": f"""
        Identifica il tipo di documento fornito. Rispondi solo con il nome del tipo di documento, senza spiegazioni aggiuntive.
        Ti verranno inviati documenti di vario tipo: fattura, email, contratto, cv, lettera, articolo, ordine d'acquisto.
        Dovrai rispondere solo con il tipo di documento, se il documento non è riconosciuto rispondi con "Altro".

        Analizza il seguente documento:
        ```
        {document_text}
        ```
        """}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)