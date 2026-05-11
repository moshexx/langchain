"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify, and resume
"""

from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: State Definitions & Factory Functions
# ============================================================

# --- 1. Approval Graph ---
class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str

def build_approval_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph that interrupts for human approval before finalization."""
    llm = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def create_draft(state: ApprovalState) -> dict:
        resp = llm.invoke(f"Create a professional response for: {state['request']}")
        return {"draft": resp.content}

    def wait_for_approval(state: ApprovalState) -> dict:
        return state

    def finalize(state: ApprovalState) -> dict:
        if state["approved"]:
            return {"final": state["draft"]}
        resp = llm.invoke(f"Revise this draft based on feedback:\n\nDraft: {state['draft']}\n\nFeedback: {state['feedback']}")
        return {"final": resp.content}

    builder = StateGraph(ApprovalState)
    builder.add_node("draft", create_draft)
    builder.add_node("approval", wait_for_approval)
    builder.add_node("finalize", finalize)
    
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approval")
    builder.add_edge("approval", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["approval"])


# --- 2. Iterative Review Graph ---
class ReviewState(TypedDict):
    document: str
    review_comments: list[str]
    revision_count: int
    status: str

def build_review_graph(model_name: str = DEFAULT_MODEL):
    """Builds a graph for iterative human review and revision cycles."""
    llm = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def submit_for_review(state: ReviewState) -> dict:
        return {"status": "pending_review"}

    def apply_feedback(state: ReviewState) -> dict:
        if not state["review_comments"]: return state
        feedback = state["review_comments"][-1]
        resp = llm.invoke(f"Revise this document based on feedback:\n\nDocument: {state['document']}\n\nFeedback: {feedback}")
        return {"document": resp.content, "revision_count": state["revision_count"] + 1, "status": "revised"}

    def route_after_review(state: ReviewState) -> Literal["apply", "done"]:
        return "done" if state["status"] == "approved" else "apply"

    def finalize(state: ReviewState) -> dict:
        return {"status": "finalized"}

    builder = StateGraph(ReviewState)
    builder.add_node("submit", submit_for_review)
    builder.add_node("apply", apply_feedback)
    builder.add_node("done", finalize)
    
    builder.add_edge(START, "submit")
    builder.add_conditional_edges("submit", route_after_review, {"apply": "apply", "done": "done"})
    builder.add_edge("apply", "submit")
    builder.add_edge("done", END)
    
    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["submit"])


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_approval_demo():
    print("=" * 60)
    print("APPROVAL WORKFLOW DEMO")
    print("=" * 60)
    app = build_approval_graph()
    config = {"configurable": {"thread_id": "demo-1"}}
    
    # Phase 1: Run until interrupt
    res = app.invoke({"request": "Interview thank-you email", "draft": "", "approved": False, "feedback": "", "final": ""}, config)
    print(f"Draft Generated. Paused for approval.\nDraft Preview: {res['draft'][:100]}...")

    # Phase 2: Update state and resume
    feedback = "Make it shorter."
    app.update_state(config, {"approved": False, "feedback": feedback})
    print(f"Human Feedback: {feedback}\nResuming...")
    
    final_res = app.invoke(None, config)
    print(f"Final Response: {final_res['final'][:150]}...\n")


def run_review_demo():
    print("=" * 60)
    print("ITERATIVE REVIEW DEMO")
    print("=" * 60)
    app = build_review_graph()
    config = {"configurable": {"thread_id": "review-1"}}
    
    # Round 0
    res = app.invoke({"document": "AI is tech.", "review_comments": [], "revision_count": 0, "status": ""}, config)
    print(f"Round 0: Paused for review. Document: {res['document']}")

    # Round 1
    feedback_1 = "Add examples."
    app.update_state(config, {"review_comments": [feedback_1], "status": "needs_revision"})
    res = app.invoke(None, config)
    print(f"Round 1 Result: {res['document'][:100]}...")

    # Round 2: Approve
    app.update_state(config, {"status": "approved"})
    final = app.invoke(None, config)
    print(f"Final Status: {final['status']} | Total Revisions: {final['revision_count']}\n")


if __name__ == "__main__":
    run_approval_demo()
    run_review_demo()
