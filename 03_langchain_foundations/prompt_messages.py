from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: Factory Functions
# ============================================================

def build_joke_prompt():
    """Returns a basic chat prompt template for jokes."""
    return ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")


def build_translation_prompt():
    """Returns a multi-message prompt template for translation."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
            ("human", "Translate the following text: {text}"),
        ]
    )


def build_few_shot_prompt():
    """Returns a few-shot chat prompt template."""
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    fewshot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
    return ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )


def build_combined_prompt():
    """Returns a combined chat prompt template."""
    system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])
    user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    return system_prompt + user_prompt


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_chat_prompt_demo():
    print("=" * 60)
    print("CHAT PROMPT TEMPLATE DEMO")
    print("=" * 60)
    prompt = build_joke_prompt()
    messages = prompt.format_messages(adjective="funny", topic="chickens")
    print(f"Formatted Messages:\n{messages}\n")


def run_multi_message_demo():
    print("=" * 60)
    print("MULTI-MESSAGE TEMPLATE DEMO")
    print("=" * 60)
    prompt = build_translation_prompt()
    messages = prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
    print(f"Formatted Messages:\n{messages}\n")


def run_message_types_demo():
    print("=" * 60)
    print("MESSAGE TYPES DEMO")
    print("=" * 60)
    messages = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there!"),
        SystemMessage(content="This is a system message."),
        ToolMessage(content="Tool executed.", tool_call_id="call_123"),
        ChatMessage(content="General message.", role="user"),
    ]
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")
    print()


def run_few_shot_demo():
    print("=" * 60)
    print("FEW-SHOT PROMPTING DEMO")
    print("=" * 60)
    prompt = build_few_shot_prompt()
    messages = prompt.format_messages(input="fast")
    print(f"Formatted Messages:\n{messages}\n")


def run_reusable_demo():
    print("=" * 60)
    print("REUSABLE/COMBINED COMPONENTS DEMO")
    print("=" * 60)
    prompt = build_combined_prompt()
    messages = prompt.format_messages(role="math expert", question="What is 2+2?")
    print(f"Formatted Messages:\n{messages}\n")


if __name__ == "__main__":
    run_chat_prompt_demo()
    run_multi_message_demo()
    run_message_types_demo()
    run_few_shot_demo()
    run_reusable_demo()
