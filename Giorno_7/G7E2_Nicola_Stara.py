from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Configurazione del modello Azure OpenAI
chat = AzureChatOpenAI(
    openai_api_base="https://<NOME-ENDPOINT>.openai.azure.com/", 
    openai_api_version="2023-05-15",
    deployment_name="<NOME-DEPLOYMENT>", 
    openai_api_key="<LA_TUA_CHIAVE_API>",
    openai_api_type="azure",
)

# Prompt per la catena
prompt = ChatPromptTemplate.from_template("Rispondi alla domanda: {question}")

# Creazione della catena
chain = LLMChain(
    llm=chat,
    prompt=prompt,
)

# Esecuzione della catena con una domanda
risposta = chain.run("Qual è la capitale della Francia?")

# Stampa della risposta
print("Risposta:", risposta)