from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.chat_models import AzureChatOpenAI # Importa la classe per Azure
from langchain.memory import ConversationBufferMemory
import os

from dotenv import load_dotenv
load_dotenv() 

# Inizializzazione modello 
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.5
)

# Inizializzazione memoria
memory = ConversationBufferMemory(return_messages=True)

# Definizione del prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un assistente amichevole e disponibile che risponde in italiano."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Creazione della catena di conversazione
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Inizio della conversazione
print("Inizia la conversazione (digita 'esci' per terminare):")

while True:
    user_input = input("Tu: ")
    if user_input.lower() == 'esci':
        break

    try:
        response = conversation.invoke({"input": user_input})
        print(f"Assistente: {response['text']}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

print("Conversazione terminata.")