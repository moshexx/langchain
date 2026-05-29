import operator
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


# ============================================================
# CORE LOGIC: State Definition & Factory Functions
# ============================================================

class ConversationState(TypedDict):
    """The state object for our graph."""
    messages: Annotated[list, operator.add]
    sentiment: str
    response_count: int


def build_conversation_graph(model_name: str = DEFAULT_MODEL):
    """Factory function that builds and compiles the conversation graph."""
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def analyze_sentiment(state: ConversationState) -> dict:
        """Node 1: Analyze the sentiment of the last message."""
        last_message = state["messages"][-1]
        response = llm.invoke(
            [
                SystemMessage(content="Classify sentiment as: positive, negative, or neutral. Reply with just the word."),
                HumanMessage(content=last_message),
            ]
        )
        return {"sentiment": response.content.lower().strip()}

    def generate_response(state: ConversationState) -> dict:
        """Node 2: Generate appropriate response based on sentiment."""
        sentiment = state["sentiment"]
        last_message = state["messages"][-1]
        
        system_prompts = {
            "positive": "Respond enthusiastically and build on their positive energy.",
            "negative": "Respond empathetically and offer support.",
            "neutral": "Respond helpfully and informatively.",
        }
        prompt = system_prompts.get(sentiment, system_prompts["neutral"])
        
        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=last_message)]
        )
        return {"messages": [f"AI: {response.content}"], "response_count": 1}

    # Construct the graph
    builder = StateGraph(ConversationState)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("generate_response", generate_response)
    
    builder.add_edge(START, "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_conversation_demo():
    print("=" * 60)
    print("FIRST GRAPH DEMO: SENTIMENT-AWARE CONVERSATION")
    print("=" * 60)
    
    app = build_conversation_graph()
    
    # Print the graph structure in ASCII
    print("\n--- Graph Structure ---")
    app.get_graph().print_ascii()
    print("-----------------------\n")
    
    # Try to save the graph visualization to a PNG file
    try:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Graph image saved to 'graph.png'\n")
    except Exception as e:
        print(f"Could not save graph image: {e}\n")
    
    test_messages = [
        "I just got promoted at work! I'm so excited!",
        "My computer crashed and I lost all my work...",
        "What's the weather like today?",
    ]

    for msg in test_messages:
        result = app.invoke({
            "messages": [f"Human: {msg}"],
            "sentiment": "",
            "response_count": 0
        })

        print(f"Input: {msg}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Response: {result['messages'][-1]}")
        print("-" * 40)


if __name__ == "__main__":
    run_conversation_demo()