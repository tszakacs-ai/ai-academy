from crewai import Agent, Task, Crew

def create_agents():
    """Crea e ritorna gli agenti Planner e Executor."""
    planner = Agent(
        role="Planner",
        goal="Analizzare i dati contrattuali",
        backstory="Esperto analista di contratti con anni di esperienza nella revisione di lavori in corso.",
        verbose=True
    )

    executor = Agent(
        role="Executor",
        goal="Generare una scheda leggibile dei contratti",
        backstory="Un comunicatore esperto che sa trasformare dati grezzi in contenuti chiari per l'utente.",
        verbose=True
    )
    return planner, executor

def create_tasks(planner, executor):
    """Crea e ritorna i task per planner e executor."""
    task_planner = Task(
        agent=planner,
        description="Trova e restituisci gli ultimi tre contratti con lavori in corso.",
        expected_output="Elenco dei contratti con i seguenti campi: cliente, nome_progetto, data_inizio."
    )

    task_executor = Task(
        agent=executor,
        description="Ricevi la lista dal planner e genera una scheda leggibile per l'utente.",
        expected_output="Testo riassuntivo per ciascuno dei tre contratti, leggibile da un utente non tecnico.",
        context=[task_planner]  # Dipendenza dal task del planner
    )

    return task_planner, task_executor

def main():
    planner, executor = create_agents()
    task_planner, task_executor = create_tasks(planner, executor)

    crew = Crew(
        agents=[planner, executor],
        tasks=[task_planner, task_executor],
        verbose=True
    )

    result = crew.run()
    print(result)

if __name__ == "__main__":
    main()
