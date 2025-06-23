import openai
from get_text import get_text


content = get_text("cleaned_output.txt")
content += "\nQual è il titolo azionario che è stato acquistato di più?"

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key=
    azure_endpoint=
    api_version=
)

response = client.chat.completions.create(
    model="o4-mini",  
    messages=[
        {"role": "system", "content": "Sei un assistente AI."},
        {"role": "user", "content": content}
    ],
    max_completion_tokens=256,
    temperature=1,
)

print(response.choices[0].message.content)