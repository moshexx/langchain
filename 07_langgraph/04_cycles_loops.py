"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

import operator
from typing import Annotated, Literal
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

# --- 1. Self-Correcting Code Gen ---
class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    success: bool

def build_self_correcting_code_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that generates and self-corrects Python code."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def generate_code(state: CodeGenState) -> dict:
        if state["iteration"] == 0:
            prompt = f"Write Python code for: {state['task']}\nReturn only the code."
        else:
            prompt = f"Fix this Python code:\n{state['code']}\n\nErrors:\n{state['errors'][-1]}\n\nReturn only code."
        
        resp = llm.invoke(prompt)
        code = resp.content.strip()
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
        return {"code": code, "iteration": state["iteration"] + 1}

    def validate_code(state: CodeGenState) -> dict:
        code = state["code"]
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"errors": [f"Syntax: {e}"], "success": False}

        # Simulating test execution (e.g. factorial)
        namespace = {}
        try:
            exec(code, namespace)
            if "solve" in namespace:
                # Add real tests here if needed
                pass
        except Exception as e:
            return {"errors": [f"Runtime: {e}"], "success": False}
        return {"success": True}

    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        if state["success"] or state["iteration"] >= state["max_iterations"]:
            return "end"
        return "generate"

    builder = StateGraph(CodeGenState)
    builder.add_node("generate", generate_code)
    builder.add_node("validate", validate_code)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", should_continue, {"generate": "generate", "end": END})
    return builder.compile()


# --- 2. Iterative Research ---
class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str

def build_iterative_research_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph for iterative research and synthesis."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def research(state: ResearchState) -> dict:
        if state["iteration"] == 0:
            query = f"Give me key facts about: {state['topic']}"
        else:
            query = f"Go deeper on {state['questions'][-1]} based on {state['findings'][-1]}"
        resp = llm.invoke(query)
        return {"findings": [resp.content]}

    def generate_questions(state: ResearchState) -> dict:
        resp = llm.invoke(f"One deeper question to explore based on: {state['findings'][-1]}")
        return {"questions": [resp.content.strip()], "iteration": state["iteration"] + 1}

    def synthesize(state: ResearchState) -> dict:
        all_findings = "\n\n".join(state["findings"])
        resp = llm.invoke(f"Synthesize these findings: {all_findings}")
        return {"summary": resp.content}

    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        return "synthesize" if state["iteration"] >= state["max_depth"] else "research"

    builder = StateGraph(ResearchState)
    builder.add_node("research", research)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "research")
    builder.add_edge("research", "generate_questions")
    builder.add_conditional_edges("generate_questions", should_continue, {"research": "research", "synthesize": "synthesize"})
    builder.add_edge("synthesize", END)
    return builder.compile()


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_code_gen_demo():
    print("=" * 60)
    print("SELF-CORRECTING CODE GEN DEMO")
    print("=" * 60)
    app = build_self_correcting_code_graph()
    res = app.invoke({"task": "recursive factorial function named 'solve'", "code": "", "errors": [], "iteration": 0, "max_iterations": 3, "success": False})
    print(f"Success: {res['success']} | Iterations: {res['iteration']}")
    print(f"Final Code:\n{res['code']}\n")


def run_research_demo():
    print("=" * 60)
    print("ITERATIVE RESEARCH DEMO")
    print("=" * 60)
    app = build_iterative_research_graph()
    res = app.invoke({"topic": "quantum computing", "findings": [], "questions": [], "iteration": 0, "max_depth": 2, "summary": ""})
    print(f"Topic: {res['topic']} | Iterations: {res['iteration']}")
    print(f"Summary Preview: {res['summary'][:200]}...\n")


if __name__ == "__main__":
    run_code_gen_demo()
    run_research_demo()
