# Query Cypher per Neo4j

cliente_fattura_query = '''
CREATE (c:Cliente {nome: "Mario Rossi"})
CREATE (f:Fattura {numero: "INV2025"})
CREATE (c)-[:HA_RICEVUTO]->(f)
'''
 
match_fattura_query = '''
MATCH (f:Fattura)
WHERE f.importo > 1000
RETURN f.numero, f.importo
'''
 
# 1. Poni una domanda sul Knowledge Graph Neo4j
domanda = "Quali clienti hanno ricevuto una fattura con importo superiore a 1000 euro?"
 
# 2. Scrivi una query Cypher per estrarre la risposta dal grafo
cypher_query = '''
MATCH (c:Cliente)-[:HA_RICEVUTO]->(f:Fattura)
WHERE f.importo > 1000
RETURN c.nome AS cliente, f.numero AS numero_fattura, f.importo AS importo
'''
 
# 3. (Simulazione) Prendi la risposta della query (formato tabellare)
risposta_query = [
    {"cliente": "Mario Rossi", "numero_fattura": "INV2025", "importo": 1500},
    {"cliente": "Luca Bianchi", "numero_fattura": "INV2026", "importo": 2000}
]
 
# 4. Passa la risposta a GPT/LLM chiedendo di riformulare in linguaggio naturale
import openai
import os
 
openai.api_key = os.getenv("OPENAI_API_KEY")  # Assicurati che la variabile sia impostata
 
def riformula_risposta(risposta_query, domanda):
    tabella = "\n".join(
        [f"- {r['cliente']} ha ricevuto la fattura {r['numero_fattura']} di importo {r['importo']} euro"
         for r in risposta_query]
    )
    prompt = (
        f"Domanda business: {domanda}\n"
        f"Risposta tabellare:\n{tabella}\n"
        "Riformula la risposta in linguaggio naturale chiaro e sintetico per un manager."
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()
 
# 5. Ottieni la risposta riformulata
risposta_naturale = riformula_risposta(risposta_query, domanda)
print(risposta_naturale)