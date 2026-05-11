"""
LangChain Core Concepts - LCEL and Runnables
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7


# ============================================================
# CORE LOGIC: Factory Functions
# ============================================================

def build_basic_chain(model_name: str = DEFAULT_MODEL):
    """Builds a basic chain using LCEL."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | model | parser


def build_translation_chain(model_name: str = DEFAULT_MODEL):
    """Builds a translation chain."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | model | parser


def build_haiku_chain(model_name: str = DEFAULT_MODEL):
    """Builds a haiku generation chain."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | model | parser


def build_summarization_chain(model_name: str = DEFAULT_MODEL):
    """Builds a summarization chain."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | model | parser


def build_marketing_chain(model_name: str = DEFAULT_MODEL):
    """Builds a marketing tagline generation chain."""
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | model | parser


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_basic_demo():
    print("=" * 60)
    print("BASIC CHAIN DEMO")
    print("=" * 60)
    chain = build_basic_chain()
    result = chain.invoke({"question": "What is LangChain?"})
    print(f"Response: {result}\n")


def run_batch_demo():
    print("=" * 60)
    print("BATCH EXECUTION DEMO")
    print("=" * 60)
    chain = build_translation_chain()
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)
    for input_data, result in zip(inputs, results):
        print(f"Input: {input_data['text']} => Output: {result}")
    print()


def run_streaming_demo():
    print("=" * 60)
    print("STREAMING DEMO")
    print("=" * 60)
    chain = build_haiku_chain()
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print("\n")


def run_schema_demo():
    print("=" * 60)
    print("SCHEMA INSPECTION DEMO")
    print("=" * 60)
    chain = build_summarization_chain()
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()
    print(f"Input Schema Variables: {list(input_schema.get('properties', {}).keys())}")
    print(f"Output Schema: {output_schema.get('type')}\n")


def run_marketing_exercise():
    print("=" * 60)
    print("EXERCISE: MARKETING TAGLINE")
    print("=" * 60)
    chain = build_marketing_chain()
    result = chain.invoke({"product": "AI agents for sales", "audience": "sales teams"})
    print(f"Marketing Tagline: {result}\n")


if __name__ == "__main__":
    # run_basic_demo()
    # run_batch_demo()
    # run_streaming_demo()
    # run_schema_demo()
    run_marketing_exercise()
