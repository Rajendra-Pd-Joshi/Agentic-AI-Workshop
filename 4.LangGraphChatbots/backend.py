from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict,Annotated
from langgraph.graph.message import BaseMessage,add_messages

class chatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

graph = StateGraph(state_schema=chatState)


llm = ChatOpenAI(model = 'gpt-4o-mini')

def chat_node(state:chatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages':[response.content]}


graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer = checkpointer)