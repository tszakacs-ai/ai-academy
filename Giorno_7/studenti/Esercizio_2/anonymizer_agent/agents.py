from crewai import Agent
from textwrap import dedent
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from tools.NER_anonymization import ner_anonymization_tool

class CustomAgents:
    def __init__(self, model_name, api_version):
        self.llm = AzureAIChatCompletionsModel(
            model_name=model_name,
            api_version=api_version
        )
        print("\nInitialized LLM for Custom Agents\n")

        print(model_name, api_version)

    def ner_agent(self):
        return Agent(
            role="Text Anonymization Specialist",
            backstory=dedent("""
                I'm an expert in Named Entity Recognition and data privacy protection.
                I've spent years removing personally identifiable information (PII) from text data
                across multiple languages and domains. I can help you anonymize sensitive information
                while preserving the context and readability of the text.
                I'm familiar with privacy regulations like GDPR, CCPA, and other data protection standards.
            """),
            goal=dedent("""
                Remove any personally identifiable information (PII) from text using advanced
                Named Entity Recognition techniques while maintaining text coherence and context.
            """),
            tools=[ner_anonymization_tool],
            allow_delegation=False,
            verbose=True,
            llm=self.llm  # Use your Azure OpenAI client
        )