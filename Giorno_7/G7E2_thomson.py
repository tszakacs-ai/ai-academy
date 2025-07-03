# langchain_azure_openai_session_new.py

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

# === CONFIG AZURE OPENAI ===
AZURE_OPENAI_API_KEY = "2UmwCov9HxZFqplz7tKqP9IoZzeiyLPrY1TfXNAKVt7G9HgWDcx1JQQJ99BFACfhMk5XJ3w3AAAAACOGPsza"
AZURE_OPENAI_ENDPOINT = "https://matth-mc4pfxah-swedencentral.cognitiveservices.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "chatgpt-thomson3"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# === Sessione LLM LangChain ===
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    temperature=1.0
)

# === Prompt Template ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un assistente AI esperto. Fornisci risposte chiare e complete."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# === Creazione Catena ===
chain_base = prompt | llm | StrOutputParser()

# === Definizione Message History Store ===
store = {}

def get_message_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# === Wrapping della catena con MessageHistory ===
chain = RunnableWithMessageHistory(
    chain_base,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

# === Esempio di sessione ===
def run_session():
    session_id = "sessione_uno"  # puoi cambiarlo per ogni sessione
    while True:
        user_input = input("\nUtente: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Sessione terminata.")
            break
        # Chiamata catena
        response = chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        print("\nAI:", response)

if __name__ == "__main__":
    print("=== Sessione AI con LangChain + Azure OpenAI ===")
    run_session()
