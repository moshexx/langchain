"""
Agent Handoffs in LangGraph
Passing control and context between agents
"""

from typing import Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: State Definitions & Factory Functions
# ============================================================

class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_reason: str
    context_summary: str

class HandoffDecision(BaseModel):
    handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(description="Agent to hand off to")
    reason: str = Field(description="Reason for handoff")
    context: str = Field(description="Key context to pass")

def build_handoff_system(model_name: str = DEFAULT_MODEL):
    """Builds a multi-agent customer service system with handoffs."""
    llm = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def triage_agent(state: HandoffState) -> dict:
        system = """Triage agent. Route to: sales, support, billing, or end."""
        decision_llm = llm.with_structured_output(HandoffDecision)
        decision = decision_llm.invoke([SystemMessage(content=system)] + state["messages"])

        if decision.handoff_to == "end":
            resp = llm.invoke([SystemMessage(content="Helpful response.")] + state["messages"])
            return {"messages": [AIMessage(content=f"[Triage] {resp.content}")], "current_agent": "end"}

        return {
            "current_agent": decision.handoff_to,
            "handoff_reason": decision.reason,
            "context_summary": decision.context,
            "messages": [AIMessage(content=f"[Triage] Routing to {decision.handoff_to}: {decision.reason}")]
        }

    def sales_agent(state: HandoffState) -> dict:
        system = f"Sales specialist. Context: {state.get('context_summary')}"
        resp = llm.invoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [AIMessage(content=f"[Sales] {resp.content}")]}

    def support_agent(state: HandoffState) -> dict:
        system = f"Support specialist. Context: {state.get('context_summary')}"
        resp = llm.invoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [AIMessage(content=f"[Support] {resp.content}")]}

    def billing_agent(state: HandoffState) -> dict:
        system = f"Billing specialist. Context: {state.get('context_summary')}"
        resp = llm.invoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [AIMessage(content=f"[Billing] {resp.content}")]}

    def route_from_triage(state: HandoffState) -> str:
        agent = state["current_agent"]
        return agent if agent in ["sales", "support", "billing"] else "end"

    builder = StateGraph(HandoffState)
    builder.add_node("triage", triage_agent)
    builder.add_node("sales", sales_agent)
    builder.add_node("support", support_agent)
    builder.add_node("billing", billing_agent)
    
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route_from_triage, {"sales": "sales", "support": "support", "billing": "billing", "end": END})
    builder.add_edge("sales", END)
    builder.add_edge("support", END)
    builder.add_edge("billing", END)
    
    return builder.compile()


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_handoff_demo():
    print("=" * 60)
    print("CUSTOMER SERVICE HANDOFF DEMO")
    print("=" * 60)
    app = build_handoff_system()
    queries = [
        "App crashes on upload",
        "Upgrade to premium",
        "Double charged on invoice",
        "What are your hours?"
    ]
    
    for q in queries:
        print(f"Customer: {q}")
        res = app.invoke({"messages": [HumanMessage(content=q)], "current_agent": "", "handoff_reason": "", "context_summary": ""})
        for m in res["messages"]:
            if isinstance(m, AIMessage): print(f"  {m.content[:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    run_handoff_demo()
