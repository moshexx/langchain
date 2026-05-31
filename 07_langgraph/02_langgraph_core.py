"""
LangGraph Core Concepts & API Reference
========================================

This file demonstrates the fundamental concepts and APIs of LangGraph:

1. StateGraph:
   - The primary class used to define stateful multi-agent and LLM workflows.
   - Initialized with a state schema (usually a TypedDict) that defines the shape and types of the shared memory (State) passed between nodes.

2. START & END:
   - START: A special control node representing the entry point of the graph. It tells the execution engine where to start (e.g., `builder.add_edge(START, "node_name")`).
   - END: A special control node representing the termination point of the graph. When execution flows into END, the graph execution halts.

3. Reducer Functions (State Aggregation):
   - By default, returning a key from a node overwrites that key in the State. Reducers allow you to specify how to *update* a key instead of overwriting it.
   - operator.add: A reducer used with `Annotated[list, operator.add]` to append elements to a list, or with `Annotated[int, operator.add]` to sum numbers.
   - add_messages: A pre-built LangGraph reducer (used in `Annotated[list[BaseMessage], add_messages]`) that appends new messages, handles updating existing messages by their ID, and simplifies chat history management.

4. Graph Builder Operations:
   - builder.add_node(name, function): Registers a state-transforming function as a node in the graph. The function receives the current State and returns a dictionary of updates to apply.
   - builder.add_edge(source, target): Defines a direct, static transition pathway from a source node to a target node.
   - builder.compile(): Validates the graph structure and compiles it into a runnable `CompiledStateGraph` instance.

5. Graph Execution & Visualizations:
   - app.invoke(inputs): Executes the compiled graph synchronously, passing the initial inputs to the state and executing nodes until reaching END.
   - app.get_graph(): Extracts the structured graph topology object.
   - app.get_graph().print_ascii(): Renders and prints a text-based ASCII flowchart of the graph's execution flow directly to the terminal.
   - app.get_graph().draw_mermaid_png(): Generates binary PNG bytes representing the graph's structure using Mermaid syntax.
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: State Definitions & Factory Functions
# ============================================================


# --- 1. Simple State ---
class SimpleState(TypedDict):
    input: str
    output: str
    step: int


def build_simple_graph():
    """Builds a basic single-node graph."""

    def process(state: SimpleState) -> dict:
        return {"output": state["input"].upper(), "step": state["step"] + 1}

    builder = StateGraph(SimpleState)
    builder.add_node("process", process)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    return builder.compile()


# --- 2. Accumulating State ---
class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: Annotated[int, operator.add]


def build_accumulating_graph():
    """Builds a graph demonstrating list/int reducers."""

    def step_one(state: AccumulatingState) -> dict:
        return {"messages": ["Step 1 executed"], "count": 1}

    def step_two(state: AccumulatingState) -> dict:
        return {"messages": ["Step 2 executed"], "count": 1}

    builder = StateGraph(AccumulatingState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile()


# --- 3. Message State ---
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_message_graph(model_name: str = DEFAULT_MODEL):
    """Builds a standard chat-based graph using add_messages."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessageState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder.compile()


# --- 4. Multi-Node Graph ---
class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str


def build_multi_node_graph(model_name: str = DEFAULT_MODEL):
    """Builds a linear 3-node graph for analysis, enhancement, and finalization."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def analyze(state: MultiStepState) -> dict:
        resp = llm.invoke(f"Summarize in one sentence: {state['input']}")
        return {"analyzed": resp.content}

    def enhance(state: MultiStepState) -> dict:
        resp = llm.invoke(f"Enhance this analysis: {state['analyzed']}")
        return {"enhanced": resp.content}

    def finalize(state: MultiStepState) -> dict:
        resp = llm.invoke(f"Finalize this into a summary: {state['enhanced']}")
        return {"final": resp.content}

    builder = StateGraph(MultiStepState)  # type: ignore
    builder.add_node("analyze", analyze)
    builder.add_node("enhance", enhance)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "enhance")
    builder.add_edge("enhance", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


# --- 5. QA Exercise Graph ---
class QAState(TypedDict):
    topic: str
    questions: str
    answer: str


def build_qa_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that generates and answers questions about a topic."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def generate_questions(state: QAState) -> dict:
        resp = llm.invoke(f"Generate 3 questions about: {state['topic']}")
        return {"questions": resp.content}

    def answer_question(state: QAState) -> dict:
        resp = llm.invoke(f"Answer the first question here:\n{state['questions']}")
        return {"answer": resp.content}

    builder = StateGraph(QAState)  # type: ignore
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("answer_question", answer_question)
    builder.add_edge(START, "generate_questions")
    builder.add_edge("generate_questions", "answer_question")
    builder.add_edge("answer_question", END)
    return builder.compile()


# ============================================================
# TEST / SIMULATION
# ============================================================


def run_simple_demo():
    print("=" * 60)
    print("SIMPLE GRAPH DEMO")
    print("=" * 60)
    app = build_simple_graph()
    result = app.invoke({"input": "hello", "output": "", "step": 0})
    print(f"Result: {result}\n")


def run_accumulating_demo():
    print("=" * 60)
    print("ACCUMULATING STATE DEMO")
    print("=" * 60)
    app = build_accumulating_graph()
    result = app.invoke({"messages": ["Initial"], "count": 0})
    print(f"Messages: {result['messages']}")
    print(f"Total Count: {result['count']}\n")


def run_message_demo():
    print("=" * 60)
    print("MESSAGE STATE DEMO")
    print("=" * 60)
    app = build_message_graph()
    result = app.invoke({"messages": [HumanMessage(content="Say Hello in Tagalog")]})
    for msg in result["messages"]:
        print(f"{type(msg).__name__}: {msg.content}")
    print()


def run_multi_node_demo():
    print("=" * 60)
    print("MULTI-NODE GRAPH DEMO")
    print("=" * 60)
    app = build_multi_node_graph()
    result = app.invoke({"input": "Artificial intelligence"})
    print(f"Final Output: {result['final']}\n")


def run_qa_exercise():
    print("=" * 60)
    print("EXERCISE: QA GRAPH")
    print("=" * 60)
    app = build_qa_graph()
    result = app.invoke({"topic": "The future of renewable energy"})
    print(f"Questions:\n{result['questions']}")
    print(f"Answer:\n{result['answer']}\n")


if __name__ == "__main__":
    # run_simple_demo()
    # run_accumulating_demo()
    # run_message_demo()
    # run_multi_node_demo()
    # run_qa_exercise()
