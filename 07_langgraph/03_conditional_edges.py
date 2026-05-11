from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: State Definitions & Factory Functions
# ============================================================

# --- 1. Router Graph ---
class RouterState(TypedDict):
    query: str
    query_type: str
    response: str

def build_router_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that routes queries by type (question, command, statement)."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def classify_query(state: RouterState) -> dict:
        resp = llm.invoke(f"Classify as 'question', 'command', or 'statement': {state['query']}")
        return {"query_type": resp.content.lower().strip()}

    def handle_question(state: RouterState) -> dict:
        resp = llm.invoke(f"Answer this question: {state['query']}")
        return {"response": f"[Answer] {resp.content}"}

    def handle_command(state: RouterState) -> dict:
        return {"response": f"[Executing] {state['query']}"}

    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged] {state['query']}"}

    def route_by_type(state: RouterState) -> Literal["question", "command", "statement"]:
        qt = state["query_type"]
        if "question" in qt: return "question"
        if "command" in qt: return "command"
        return "statement"

    builder = StateGraph(RouterState)
    builder.add_node("classify", classify_query)
    builder.add_node("handle_question", handle_question)
    builder.add_node("handle_command", handle_command)
    builder.add_node("handle_statement", handle_statement)
    
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_type, {
        "question": "handle_question",
        "command": "handle_command",
        "statement": "handle_statement"
    })
    builder.add_edge("handle_question", END)
    builder.add_edge("handle_command", END)
    builder.add_edge("handle_statement", END)
    
    return builder.compile()


# --- 2. Quality Loop Graph ---
class QualityState(TypedDict):
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int

def build_quality_loop_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that iterates until quality score is high enough."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def evaluate_quality(state: QualityState) -> dict:
        resp = llm.invoke(f"Rate quality 1-10 (number only): {state['content']}")
        try: score = int(resp.content.strip())
        except: score = 5
        return {"quality_score": score}

    def improve_content(state: QualityState) -> dict:
        resp = llm.invoke(f"Improve this content: {state['content']}")
        return {"content": resp.content, "iteration": state["iteration"] + 1}

    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state["content"],
            "feedback": f"Approved iteration {state['iteration']} (Score: {state['quality_score']})"
        }

    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 7 or state["iteration"] >= 3:
            return "finalize"
        return "improve"

    builder = StateGraph(QualityState)
    builder.add_node("evaluate", evaluate_quality)
    builder.add_node("improve", improve_content)
    builder.add_node("finalize", finalize_content)
    
    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", should_continue, {"improve": "improve", "finalize": "finalize"})
    builder.add_edge("improve", "evaluate")
    builder.add_edge("finalize", END)
    
    return builder.compile()


# --- 3. Multi-Path Task Graph ---
class TaskState(TypedDict):
    task: str
    urgency: str
    complexity: str
    handler: str
    result: str

def build_task_routing_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that routes based on urgency and complexity."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def analyze_task(state: TaskState) -> dict:
        u_resp = llm.invoke(f"Is this 'urgent' or 'normal'? {state['task']}")
        c_resp = llm.invoke(f"Is this 'complex' or 'simple'? {state['task']}")
        return {"urgency": u_resp.content.lower().strip(), "complexity": c_resp.content.lower().strip()}

    def route_task(state: TaskState) -> str:
        u, c = "urgent" in state["urgency"], "complex" in state["complexity"]
        if u and c: return "urgent_complex"
        if u: return "urgent_simple"
        if c: return "normal_complex"
        return "normal_simple"

    builder = StateGraph(TaskState)
    builder.add_node("analyze", analyze_task)
    builder.add_node("urgent_complex", lambda x: {"handler": "Senior", "result": "Escalated"})
    builder.add_node("urgent_simple", lambda x: {"handler": "Quick", "result": "Handled fast"})
    builder.add_node("normal_complex", lambda x: {"handler": "Specialist", "result": "Assigned"})
    builder.add_node("normal_simple", lambda x: {"handler": "Standard", "result": "Queued"})

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges("analyze", route_task, {
        "urgent_complex": "urgent_complex",
        "urgent_simple": "urgent_simple",
        "normal_complex": "normal_complex",
        "normal_simple": "normal_simple"
    })
    for node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        builder.add_edge(node, END)
    
    return builder.compile()


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_router_demo():
    print("=" * 60)
    print("BASIC ROUTING DEMO")
    print("=" * 60)
    app = build_router_graph()
    queries = ["What is the capital of France?", "Send an email", "I love AI"]
    for q in queries:
        res = app.invoke({"query": q})
        print(f"Q: {q} | Type: {res['query_type']} | Resp: {res['response']}\n")


def run_loop_demo():
    print("=" * 60)
    print("CONDITIONAL LOOP DEMO")
    print("=" * 60)
    app = build_quality_loop_graph()
    res = app.invoke({"content": "AI is cool", "quality_score": 0, "feedback": "", "final_content": "", "iteration": 0})
    print(f"Final: {res['final_content'][:100]}...\n{res['feedback']}\n")


def run_task_demo():
    print("=" * 60)
    print("MULTI-PATH ROUTING DEMO")
    print("=" * 60)
    app = build_task_routing_graph()
    tasks = ["Server down!", "Fix typo"]
    for t in tasks:
        res = app.invoke({"task": t})
        print(f"Task: {t} | Handler: {res['handler']} | Result: {res['result']}\n")


if __name__ == "__main__":
    run_router_demo()
    run_loop_demo()
    run_task_demo()
