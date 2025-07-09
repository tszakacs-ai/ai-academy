from ..utils import safe_gpt_call
from .ai_client import AIProjectClientDefinition


class ChatCompletionModel(AIProjectClientDefinition):
    """Wrapper per i modelli di chat Azure OpenAI."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(
            api_version="2025-01-01-preview"
        )

    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Sei un assistente AI specializzato in analisi di documenti testuali."
            },
            {"role": "user", "content": f"Documento:\n{content}\n\nDomanda: {question}"},
        ]
        response = safe_gpt_call(
            self.azure_client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content
