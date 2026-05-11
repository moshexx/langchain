"""
Checkpointing and Persistence in LangGraph
Save and resume agent state
"""

import operator
import tempfile
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DB_PATH = "./langgraph_checkpoints.db"


# ============================================================
# CORE LOGIC: State Definitions & Factory Functions
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


def build_chat_graph(model_name: str = DEFAULT_MODEL, checkpointer=None):
    """Builds a basic chat graph with optional checkpointing."""
    llm = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def chat(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


class TaskState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    step: str

def build_task_graph(model_name: str = DEFAULT_MODEL, checkpointer=None):
    """Builds a 2-node task graph (analyze -> summarize) with optional checkpointing."""
    llm = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def analyze(state: TaskState) -> dict:
        resp = llm.invoke(state["messages"])
        return {"messages": [resp], "step": "analyzed"}

    def summarize(state: TaskState) -> dict:
        resp = llm.invoke([HumanMessage(content=f"Summarize in one sentence: {state['messages'][-1].content}")])
        return {"messages": [resp], "step": "summarized"}

    builder = StateGraph(TaskState)
    builder.add_node("analyze", analyze)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_memory_demo():
    print("=" * 60)
    print("MEMORY SAVER DEMO (In-Memory Checkpointing)")
    print("=" * 60)
    app = build_chat_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-1"}}
    
    res1 = app.invoke({"messages": [HumanMessage(content="My name is Paulo")]}, config)
    print(f"Turn 1: {res1['messages'][-1].content}")
    
    res2 = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
    print(f"Turn 2: {res2['messages'][-1].content}\n")


def run_sqlite_demo():
    print("=" * 60)
    print("SQLITE PERSISTENCE DEMO")
    print("=" * 60)
    
    with SqliteSaver.from_conn_string(DB_PATH) as saver:
        app = build_chat_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "persistent-thread"}}
        
        app.invoke({"messages": [HumanMessage(content="The secret is 1234")]}, config)
        print("Session 1: Secret stored.")
        
        res = app.invoke({"messages": [HumanMessage(content="What was the secret?")]}, config)
        print(f"Session 2: {res['messages'][-1].content}\n")


def run_inspection_demo():
    print("=" * 60)
    print("STATE INSPECTION DEMO")
    print("=" * 60)
    app = build_chat_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "inspect-me"}}
    
    app.invoke({"messages": [HumanMessage(content="Message 1")]}, config)
    app.invoke({"messages": [HumanMessage(content="Message 2")]}, config)
    
    state = app.get_state(config)
    print(f"Next Node: {state.next}")
    print(f"Messages count: {len(state.values['messages'])}")
    
    print("\nHistory:")
    for i, snapshot in enumerate(app.get_state_history(config)):
        print(f"  Snapshot {i}: {len(snapshot.values['messages'])} messages")
    print()


def run_branching_demo():
    print("=" * 60)
    print("BRANCHING CONVERSATIONS DEMO")
    print("=" * 60)
    app = build_chat_graph(checkpointer=MemorySaver())
    
    main_config = {"configurable": {"thread_id": "main"}}
    app.invoke({"messages": [HumanMessage(content="Let's talk about travel.")]}, main_config)
    main_state = app.get_state(main_config)
    
    # Branch A
    branch_a = {"configurable": {"thread_id": "beach"}}
    app.update_state(branch_a, main_state.values)
    res_a = app.invoke({"messages": [HumanMessage(content="Beach or Mountains?")]}, branch_a)
    print(f"Branch Beach: {res_a['messages'][-1].content[:50]}...")

    # Branch B
    branch_b = {"configurable": {"thread_id": "space"}}
    app.update_state(branch_b, main_state.values)
    res_b = app.invoke({"messages": [HumanMessage(content="Is space travel real?")]}, branch_b)
    print(f"Branch Space: {res_b['messages'][-1].content[:50]}...\n")


def run_internals_demo():
    print("=" * 60)
    print("CHECKPOINT INTERNALS DEMO")
    print("=" * 60)
    app = build_task_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "internals"}}
    
    app.invoke({"messages": [HumanMessage(content="Blue sky explanation")], "step": ""}, config)
    
    state = app.get_state(config)
    print(f"Value 'step': {state.values['step']}")
    print(f"Config: {state.config}")
    print(f"Created At: {state.created_at}")
    print()


if __name__ == "__main__":
    run_memory_demo()
    run_sqlite_demo()
    run_inspection_demo()
    run_branching_demo()
    run_internals_demo()
