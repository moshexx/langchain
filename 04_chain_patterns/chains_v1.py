"""
Understanding Chains in LangChain V.1
LCEL patterns, composition, and debugging
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: Factory Functions
# ============================================================

def build_basic_summary_chain(model_name: str = DEFAULT_MODEL):
    """Builds a basic summary chain."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    model = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return prompt | model | StrOutputParser()


def build_parallel_analysis_chain(model_name: str = DEFAULT_MODEL):
    """Builds a parallel chain for summary, keywords, and sentiment."""
    model = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()

    summary_chain = ChatPromptTemplate.from_template("Summarize in two sentences: {text}") | model | parser
    keywords_chain = ChatPromptTemplate.from_template("Extract 5 keywords from: {text}\nReturn as comma-separated list.") | model | parser
    sentiment_chain = ChatPromptTemplate.from_template("What is the sentiment of: {text}") | model | parser

    return RunnableParallel(summary=summary_chain, keywords=keywords_chain, sentiment=sentiment_chain)


def build_retrieval_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain with passthrough and a simulated retriever."""
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\nContext: {context}\n\nAnswer the question based on the context."
    )
    model = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def fake_retriever(input_dict):
        return "LangChain was created by Harrison Chase in 2022."

    return (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough()
        )
        | RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]["question"]})
        | prompt | model | StrOutputParser()
    )


def build_branching_chain(model_name: str = DEFAULT_MODEL):
    """Builds a branching chain for code vs general questions."""
    model = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    
    classifier_prompt = ChatPromptTemplate.from_template("Classify this as 'code' or 'general': {input}")
    classifier = classifier_prompt | model | StrOutputParser()

    code_prompt = ChatPromptTemplate.from_template("You are a coding expert. Help with: {input}")
    general_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer: {input}")

    def is_code_question(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()

    return RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser()
    )


def build_debug_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain with intermediate logging steps for debugging."""
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    model = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)

    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    return (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_basic_demo():
    print("=" * 60)
    print("BASIC CHAIN DEMO")
    print("=" * 60)
    chain = build_basic_summary_chain()
    result = chain.invoke({"text": "LangChain is a framework for developing LLM applications."})
    print(f"Summary: {result}\n")


def run_parallel_demo():
    print("=" * 60)
    print("PARALLEL CHAIN DEMO")
    print("=" * 60)
    chain = build_parallel_analysis_chain()
    text = "The new AI features are incredible! Users love the speed, but pricing is high. Success is clear."
    results = chain.invoke({"text": text})
    print(f"Summary: {results['summary']}")
    print(f"Keywords: {results['keywords']}")
    print(f"Sentiment: {results['sentiment']}\n")


def run_passthrough_demo():
    print("=" * 60)
    print("PASSTHROUGH DEMO")
    print("=" * 60)
    chain = build_retrieval_chain()
    result = chain.invoke({"question": "Who created LangChain?"})
    print(f"Answer: {result}\n")


def run_branching_demo():
    print("=" * 60)
    print("BRANCHING DEMO")
    print("=" * 60)
    chain = build_branching_chain()
    questions = ["How do I write a for loop in Python?", "What's the weather?"]
    for q in questions:
        result = chain.invoke({"input": q})
        print(f"Q: {q}\nA: {result[:100]}...\n")


def run_debugging_demo():
    print("=" * 60)
    print("DEBUGGING DEMO")
    print("=" * 60)
    chain = build_debug_chain()
    result = chain.invoke({"name": "DebugUser"})
    print(f"Final Greeting: {result}\n")


if __name__ == "__main__":
    run_basic_demo()
    run_parallel_demo()
    run_passthrough_demo()
    run_branching_demo()
    run_debugging_demo()
