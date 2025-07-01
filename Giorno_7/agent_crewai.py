from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

import os 
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o_deploy",
    temperature=0,
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model_kwargs={
        "api-version": "2025-01-01-preview"  
    }
)

agent = Agent(
    role="Esperto di Privacy",
    goal="Anonimizzare documenti rimuovendo PII",
    backstory="Un consulente GDPR esperto nella protezione dei dati.",
    llm=llm,
    verbose=True
)

task = Task(
    description="Rimuovi o sostituisci tutti i dati personali (nome, email, CF, indirizzo) nel testo.",
    agent=agent,
    expected_output="Testo anonimizzato."

)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

documento = """
Nome: Laura Neri
Codice Fiscale: NRALRA80B01H501X
Email: laura.neri@example.com
Indirizzo: Via Milano 10, Roma
"""

risultato = crew.kickoff(inputs={"input": documento})
print(risultato)
