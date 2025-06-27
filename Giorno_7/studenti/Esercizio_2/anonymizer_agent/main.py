import os
from crewai import Crew
from agents import CustomAgents
from tasks import CustomTasks
from dotenv import load_dotenv
from pathlib import Path

class AnonymizationCrew:
    def __init__(self, text_to_anonymize, model_name, api_version):
        self.text_to_anonymize = text_to_anonymize
        self.model_name = model_name
        self.api_version = api_version

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = CustomAgents(self.model_name, self.api_version)
        tasks = CustomTasks()

        # Define your custom agents and tasks here
        custom_agent_1 = agents.ner_agent()

        # Custom tasks include agent name and variables as input
        custom_task_1 = tasks.create_anonymization_task(
            custom_agent_1,
            self.text_to_anonymize,
        )

        # Define your custom crew here
        crew = Crew(
            agents=[custom_agent_1],
            tasks=[custom_task_1],
            verbose=True,
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    
    load_dotenv()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Set environment variables that LangChain expects
    os.environ["AZURE_INFERENCE_CREDENTIAL"] = api_key
    os.environ["AZURE_INFERENCE_ENDPOINT"] = azure_endpoint

    local_dir = Path(__file__).resolve().parent
    with open(local_dir / "sample_data.csv", "r", encoding="utf-8") as f:
        text_to_anonymize = f.readlines()[1:]

    custom_crew = AnonymizationCrew(text_to_anonymize, deployment_name, api_version)
    result = custom_crew.run()

    print("\n\n########################")
    print("## Here is you custom crew run result:")
    print("########################\n")

    print(result)