"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, streaming, and cost optimization
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
COMPARISON_MODELS = ["gpt-4o-mini", "gpt-4o"]


# ============================================================
# CORE LOGIC: Factory Functions
# ============================================================

def build_chat_model(model_name: str = DEFAULT_MODEL, temperature: float = 0.7, streaming: bool = True):
    """Initializes a chat model using init_chat_model."""
    return init_chat_model(
        model=model_name,
        temperature=temperature,
        streaming=streaming,
        max_retries=3,
    )


def get_multi_model_responses(question: str, model_names: list[str]) -> dict[str, str]:
    """Gets responses from multiple models for a single question."""
    responses = {}
    for model_name in model_names:
        model = build_chat_model(model_name=model_name, streaming=False)
        response = model.invoke(question)
        responses[model_name] = response.content
    return responses


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_init_model_demo():
    print("=" * 60)
    print("INIT CHAT MODEL DEMO")
    print("=" * 60)
    chat_model = build_chat_model()
    response = chat_model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response: {response.content}\n")


def run_comparison_demo():
    print("=" * 60)
    print("MODEL COMPARISON DEMO")
    print("=" * 60)
    prompt = "Explain recursion in one sentence."
    print(f"Prompt: {prompt}\n")
    
    results = get_multi_model_responses(prompt, COMPARISON_MODELS)
    for model_name, answer in results.items():
        print(f"Response from {model_name}: {answer}\n")


def run_message_demo():
    print("=" * 60)
    print("MESSAGE OBJECTS & MULTI-TURN DEMO")
    print("=" * 60)
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    
    # turn 1
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    response = model.invoke(messages)
    print(f"Pirate: {response.content}\n")

    # turn 2
    messages.append(response)
    messages.append(HumanMessage(content="What about tomorrow?"))
    follow_up = model.invoke(messages)
    print(f"Pirate (Follow-up): {follow_up.content}\n")


def run_multi_model_exercise():
    print("=" * 60)
    print("EXERCISE: MULTI-MODEL SETUP")
    print("=" * 60)
    results = get_multi_model_responses("What is AI?", ["gpt-4o-mini", "gpt-4o"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer[:100]}...\n")


if __name__ == "__main__":
    run_init_model_demo()
    run_comparison_demo()
    run_message_demo()
    run_multi_model_exercise()
