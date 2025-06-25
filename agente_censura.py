from crewai import Agent, Task, Crew
from crewai_tools import (DirectoryReadTool,FileReadTool,SerperDevTool,WebsiteSearchTool)
from crewai.llm import LLM

dir_tool= DirectoryReadTool(directory="./Giorno_7")
file_tool= FileReadTool()

files = ["./Mail.txt"]

llm = LLM(
    model="azure/gpt-4.1",
    api_version="2024-12-01-preview",
    api_key="API-KEY", 
    api_base="API-ENDPOINT"
)


# Definizione dell’agente Planner CrewAI
planner = Agent(
    llm=llm,
    type="planner",
    name="Anonimizzatore",
    role="Anonimizza documenti",
    goal="Identificare i dati sensibili (IBAN, email, nome e cognome) nei documenti e sostituirli con un placeholder",
    backstory="Esperto nella protezione dei dati personali, specializzato nell'anonimizzazione di documenti aziendali.",
    tools=[dir_tool, file_tool]
)

task_planner = Task(
    agent=planner,
    description="Apri i file nella cartella specificata e identifica i dati sensibili da anonimizzare",
    expected_output='Documenti anonimizzati'
)

crew= Crew(
    agents=[planner],
    tasks=[task_planner])

result = crew.kickoff()
print(result)