from crewai import Task
from textwrap import dedent


class CustomTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def create_anonymization_task(self, agent, text_to_anonymize: str):

        print("\nCreated a task for the NER agent to anonymize text\n")

        return Task(
            description=dedent(f"""
                Anonymize the following text by identifying and replacing all personally 
                identifiable information (PII) including names, locations, organizations, 
                and other sensitive entities:
                
                Text to anonymize: {text_to_anonymize}
                
                Use the NER anonymization tool to process the text and return the anonymized version.
                Ensure that the anonymized text maintains its original meaning and context while 
                protecting privacy.

                {self.__tip_section()}
            """),
            agent=agent,
            expected_output="Anonymized text with all PII replaced by appropriate placeholders"
        )
