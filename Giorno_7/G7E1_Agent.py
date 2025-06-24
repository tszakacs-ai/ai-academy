from crewai import Agent, Task, Crew
from crewai_tools import (DirectoryReadTool,FileReadTool,SerperDevTool,WebsiteSearchTool)
from crewai.llm import LLM

dir_tool= DirectoryReadTool(directory="./Giorno_6")
file_tool= FileReadTool()

llm = LLM(
    model="azure/gpt-4.1",
    api_version="2025-01-01-preview",
    api_key='',
    api_base='https://lorenzomartemucci-academy.openai.azure.com/',
)

# Definizione dell’agente Planner CrewAI
anonymizer_agent = Agent(
    llm=llm,
    type="planner",
    name="Anonimizzatore",
    role="Anonimizza documenti",
    goal="Identificare i dati sensibili tipo IBAN, email, nome e cognome nei documenti e sostituiscili con un placeholder",
    backstory="Esperto nella protezione dei dati personali, specializzato nell'anonimizzazione di documenti aziendali.",
    tools=[dir_tool, file_tool]
)

task_agent = Task(
    agent=anonymizer_agent,
    description="Apri i file nella cartella specificata e identifica i dati sensibili da anonimizzare",
    expected_output='Documenti anonimizzati'
)

crew= Crew(
    agents=[anonymizer_agent],
    tasks=[task_agent])

result = crew.kickoff()
print(result)