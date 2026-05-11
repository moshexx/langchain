"""
Prompt Templates and Messages in LangChain V.1
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: Factory Functions
# ============================================================

def build_basic_prompt():
    """Returns a simple translation prompt template."""
    return ChatPromptTemplate.from_template("Translate '{text}' to {language}")


def build_multi_prompt():
    """Returns a multi-message translation prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Be concise."),
            ("human", "Translate '{text}' to {language}"),
        ]
    )


def build_placeholder_prompt():
    """Returns a prompt template with a MessagesPlaceholder."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


def build_few_shot_prompt():
    """Returns a few-shot prompt template for word opposites."""
    examples = [
        {"word": "happy", "opposite": "sad"},
        {"word": "fast", "opposite": "slow"},
        {"word": "big", "opposite": "small"},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "What's the opposite of '{word}'?"),
            ("ai", "The opposite of '{word}' is '{opposite}'."),
        ]
    )
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You give the opposite of words. Follow the examples."),
            few_shot,
            ("human", "What's the opposite of '{word}'?"),
        ]
    )


def build_composed_prompt():
    """Returns a composed prompt template (Persona + Task)."""
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )
    task = ChatPromptTemplate.from_messages([("human", "{task}")])
    return persona + task


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_basic_demo():
    print("=" * 60)
    print("BASIC TEMPLATES DEMO")
    print("=" * 60)
    prompt = build_basic_prompt()
    messages = prompt.format_messages(text="Hello, world!", language="French")
    print(f"Simple template messages: {messages}\n")


def run_message_types_demo():
    print("=" * 60)
    print("MESSAGE TYPES DEMO (Conversation)")
    print("=" * 60)
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    messages = [
        SystemMessage(content="You are a math tutor. Be brief."),
        HumanMessage(content="What's 5 * 5?"),
        AIMessage(content="25"),
        HumanMessage(content="And if I add 10?"),
    ]
    response = model.invoke(messages)
    print(f"Conversation result: {response.content}\n")


def run_placeholder_demo():
    print("=" * 60)
    print("MESSAGES PLACEHOLDER DEMO")
    print("=" * 60)
    prompt = build_placeholder_prompt()
    history = [
        HumanMessage(content="My name is Paulo"),
        AIMessage(content="Nice to meet you, Paulo!"),
    ]
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    chain = prompt | model
    response = chain.invoke({"history": history, "question": "What's my name?"})
    print(f"Response: {response.content}\n")


def run_few_shot_demo():
    print("=" * 60)
    print("FEW-SHOT DEMO")
    print("=" * 60)
    prompt = build_few_shot_prompt()
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    chain = prompt | model
    response = chain.invoke({"word": "bright"})
    print(f"Few-shot result: {response.content}\n")


def run_composition_demo():
    print("=" * 60)
    print("PROMPT COMPOSITION DEMO")
    print("=" * 60)
    prompt = build_composed_prompt()
    model = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7)
    chain = prompt | model
    
    print("Persona: Pirate Captain")
    pirate_resp = chain.invoke({"role": "pirate captain", "tone": "adventurous", "task": "Tell me about your ship"})
    print(f"Pirate Response: {pirate_resp.content[:100]}...\n")

    print("Persona: Scientist")
    scientist_resp = chain.invoke({"role": "scientist", "tone": "precise", "task": "Explain gravity"})
    print(f"Scientist Response: {scientist_resp.content[:100]}...\n")


if __name__ == "__main__":
    run_basic_demo()
    run_message_types_demo()
    run_placeholder_demo()
    run_few_shot_demo()
    run_composition_demo()
