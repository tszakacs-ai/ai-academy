import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import Annotated, List
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import AzureChatOpenAI

load_dotenv()
 
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
 
response = llm.invoke("Qual è la capitale dell'Italia?")
print(response.content)