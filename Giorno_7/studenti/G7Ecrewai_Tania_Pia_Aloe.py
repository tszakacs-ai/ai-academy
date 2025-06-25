from crewai import Agent, Task, Crew
import re
 
# Funzione manuale per anonimizzare
def anonimizza_contratti(contratti):
    def anonimizza_nome(nome):
        iniziali = re.findall(r'\b(\w)', nome)
        return ' '.join([f"{i.upper()}." for i in iniziali])
 
    def anonimizza_iban(iban):
        return re.sub(r'^(.{4})(.*)(.{4})$', lambda m: m.group(1) + '*' * len(m.group(2)) + m.group(3), iban)
 
    return [
        {
            "nome": anonimizza_nome(c["nome"]),
            "iban": anonimizza_iban(c["iban"]),
            "data": c["data"]
        }
        for c in contratti
    ]
 
# Lista di esempio
contratti = [
    {"nome": "Mario Rossi", "iban": "IT60X0542811101000000123456", "data": "2024-05-10"},
    {"nome": "Giulia Bianchi", "iban": "IT22A1234567890000001234567", "data": "2024-06-05"},
    {"nome": "Luigi Verdi", "iban": "IT99Z0987654321000001234567", "data": "2024-06-15"}
]
 
# Agente semplice (senza tool)
anon_agent = Agent(
    name="AnonimizzatoreContratti",
    role="Esperto privacy",
    goal="Anonimizzare nome e IBAN dei contratti",
    backstory="Lavora nel dipartimento legale per protezione dati",
)
 
# Task con input predefinito
anon_task = Task(
    description="Ricevi i contratti e restituisci una lista anonimizzata con iniziali e IBAN mascherato",
    expected_output="Lista anonimizzata",
    agent=anon_agent,
    input_data=contratti
)
 
# Crew con 1 solo agente e task
crew = Crew(
    agents=[anon_agent],
    tasks=[anon_task],
    verbose=True
)
 
# Esecuzione "manuale"
print("Contratti anonimizzati:\n")
risultato = anonimizza_contratti(contratti)
for c in risultato:
    print(c)