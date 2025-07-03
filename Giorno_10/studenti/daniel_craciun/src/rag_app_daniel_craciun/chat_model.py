from .ai_client import AIProjectClientDefinition

class ChatCompletionModel:
    def __init__(
        self,
        ai_client: AIProjectClientDefinition,
        model_name: str = None,
        system_prompt: str = None,
    ):
        self.client = ai_client.client
        self.model_name = model_name or "gpt-4o"
        self.system_prompt = system_prompt or "You are an AI assistant for text document analysis."

    def ask_about_document(self, content: str, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Document:\n{content}\n\nQuestion: {question}"},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
        )
        return response.choices[0].message.content
