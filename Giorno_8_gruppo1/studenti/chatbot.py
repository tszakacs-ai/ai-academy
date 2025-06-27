import os
import json
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ChatbotAnalyzer:
    """
    Class for analyzing text using Azure OpenAI models via LangChain.
    Can produce a summary, a semantic analysis, and a formal customer reply.
    """

    def __init__(
        self,
        dotenv_path: str,
        deployment_name: str = "gpt-4o",
        use_combined_prompt: bool = False
    ):
        """
        Initializes the Azure OpenAI model and configures prompts.

        Args:
            dotenv_path (str): Path to the .env file containing Azure credentials.
            deployment_name (str): Azure OpenAI deployment name.
            use_combined_prompt (bool): If True, uses a single prompt for all analyses.
        """
        # Load environment variables from .env file
        load_dotenv(dotenv_path=dotenv_path)

        # Initialize Azure OpenAI chat model
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_base=os.getenv("AZURE_ENDPOINT_4o"),
            openai_api_version="2024-12-01-preview",
            openai_api_key=os.getenv("AZURE_API_KEY_4o")
        )

        self.use_combined_prompt = use_combined_prompt
        self._build_prompts()  # Build prompts to use

    def _build_prompts(self):
        """
        Builds LangChain prompts based on the selected mode.
        """
        if self.use_combined_prompt:
            # Single prompt requesting summary, analysis, and reply
            combined_template = """
                L'azienda si chiama "SmartDocs Srl".
                Dato il seguente testo, esegui:
                1. Un riassunto conciso
                2. Un'analisi semantica dettagliata
                3. Una risposta formale al cliente

                Testo:
                {text}

                Scrivi tutto separato da intestazioni come:
                === Riassunto ===, === Analisi Semantica ===, === Risposta al Cliente ===

                Non fare riferimento a placeholder o entità anonimizzate.
                Usa un tono professionale e completo.
                Non fare riferimento ai dati anonimizzati come nome ed iban 
                Non riportare i dati anonimizzati come [NAME] o [IBAN] nella mail originale e non dedurrli dal testo.
                Non includere nella risposta placeholder per i dati anonimi.
            """
            combined_prompt = PromptTemplate(input_variables=["text"], template=combined_template)
            self.combined_chain = LLMChain(llm=self.llm, prompt=combined_prompt)

        else:
            # Separate prompts for each type of analysis
            self.summary_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
                input_variables=["text"],
                template="Provide a concise summary of the following text:\n\n{text}"
            ))
            self.analysis_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
                input_variables=["text"],
                template="Provide a detailed semantic analysis of the following text:\n\n{text}"
            ))
            self.reply_chain = LLMChain(llm=self.llm, prompt=PromptTemplate(
                input_variables=["text"],
                template="Prepare a formal reply to the customer based on this text:\n\n{text}"
            ))

    def process_text(self, text: str) -> dict:
        """
        Processes the provided text and produces the desired results.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary with the key 'full_output' or
                  'summary', 'analysis', and 'reply'.
        """
        if not text.strip():
            raise ValueError("The provided text is empty.")

        if self.use_combined_prompt:
            output = self.combined_chain.run(text=text)
            return {"full_output": output}
        else:
            return {
                "summary": self.summary_chain.run(text=text),
                "analysis": self.analysis_chain.run(text=text),
                "reply": self.reply_chain.run(text=text)
            }

# === DIRECT TERMINAL USAGE ===
if __name__ == "__main__":
    # Path to the .env file with keys and endpoints
    dotenv_path = r"C:\Users\BG726XR\OneDrive - EY\Desktop\academy_profice\ai-academy-1\.env"
    
    # Initialize the analyzer (True = unified prompt, False = separate)
    analyzer = ChatbotAnalyzer(
        dotenv_path=dotenv_path,
        use_combined_prompt=True
    )

    # Request input from terminal
    print("Inserisci testo anonimo:")
    text = input(">> ")

    # Run the analysis
    results = analyzer.process_text(text)

    # Output the results
    print("\n=== RESULTS ===\n")
    if "full_output" in results:
        print(results["full_output"])
    else:
        print("\n=== Summary ===\n", results["summary"])
        print("\n=== Semantic Analysis ===\n", results["analysis"])
        print("\n=== Customer Reply ===\n", results["reply"])
