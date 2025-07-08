from typing import Annotated
from langchain_core.messages import AIMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from client import instanziate_client 

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    id : str
    messages: Annotated[list, add_messages]

def to_openai_format(messages):
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system"
    }
    return [{"role": role_map.get(m.type, m.type), "content": m.content} for m in messages]

def chatbot(state: State):
    client = instanziate_client()
    openai_msgs = to_openai_format(state["messages"])
    print("OpenAI messages:", openai_msgs)
 
    response = client.chat.completions.create(
        model="gpt-4o_deploy",
        messages=openai_msgs
    )
 
    content = response.choices[0].message.content
    return {"messages": [AIMessage(content=content)]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_edge("chatbot", END)
 
app = graph_builder.compile()

def stream_graph_updates(user_input: str):
    initial_state = {
        "messages": [
            {"role": "system", "content": "Sei un assistente intelligente."},
            {"role": "user", "content": user_input}
        ]
    }
    print("🔍 Stato iniziale:", initial_state)
    for event in app.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
