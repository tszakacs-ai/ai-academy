from crewai import Agent, Task, Crew
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai.llm import LLM

def setup_tools():
    """Inizializza e ritorna gli strumenti necessari."""
    dir_tool = DirectoryReadTool(directory="./Giorno_6")
    file_tool = FileReadTool()
    return [dir_tool, file_tool]

def setup_llm():
    """Configura e ritorna il modello LLM."""
    return LLM(
        model="azure/gpt-4.1",
        api_version="2025-01-01-preview",
        api_key='',         # Inserisci qui la tua API key
        api_base=''         # Inserisci qui il tuo endpoint API
    )

def create_agent(llm, tools):
    """Crea e ritorna l’agente Planner per l’anonimizzazione."""
    return Agent(
        llm=llm,
        type="planner",
        name="Anonimizzatore",
        role="Anonimizza documenti",
        goal="Identificare i dati sensibili tipo IBAN, email, nome e cognome nei documenti e sostituiscili con un placeholder",
        backstory="Esperto nella protezione dei dati personali, specializzato nell'anonimizzazione di documenti aziendali.",
        tools=tools
    )

def main():
    tools = setup_tools()
    llm = setup_llm()

    planner = create_agent(llm, tools)

    task_planner = Task(
        agent=planner,
        description="Apri i file nella cartella specificata e identifica i dati sensibili da anonimizzare",
        expected_output='Documenti anonimizzati'
    )

    crew = Crew(
        agents=[planner],
        tasks=[task_planner]
    )

    result = crew.kickoff()
    print(result)

if __name__ == "__main__":
    main()
