import openai
import pathlib

def get_text(file_name):
    with open(pathlib.Path(__file__).parent / file_name, "r", encoding="utf-8") as file:
        text = file.read()
        return text.strip() if text else "No content found in the file."

content = get_text("cleaned_output.txt")
content += "\nQual è il titolo azionario che è stato acquistato di più?"

# Crea il client Azure OpenAI
client = openai.AzureOpenAI(
    api_key="your_api_key_here",  
    azure_endpoint="your_azure_endpoint_here",  
    api_version="your_api_version_here"  
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