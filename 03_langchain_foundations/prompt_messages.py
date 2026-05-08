from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# import os
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ChatMessage,
    ToolMessage,
)
from langchain_core.prompts import FewShotChatMessagePromptTemplate

load_dotenv()


def demo_chat_prompt_template() -> None:
    """Demonstrates basic ChatPromptTemplate usage."""
    # Basic template
    prompt = ChatPromptTemplate.from_template(
        "Tell me a {adjective} joke about {topic}."
    )

    # Format and inspect
    messages = prompt.format_messages(adjective="funny", topic="chickens")
    print("Basic Template Output:")
    print(messages)
    print()


def demo_multi_message_templates():
    """Demonstrates multi-message templates with system and human roles."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "Translate the following text: {text}"),
        ]
    )

    messages = prompt.format_messages(
        input_language="English", output_language="French", text="I love programming."
    )
    print("Multi-Message Template Output:")
    print(messages)

    # model = init_chat_model(model="gpt-4o-mini", temperature=0)
    # response = model.invoke(messages)
    # print(response.content)
    print()


def demo_message_types():
    """Demonstrates the different message types available in LangChain."""
    messages = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I assist you today?"),
        SystemMessage(content="This is a system message."),
        ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
        ChatMessage(content="This is a general chat message.", role="user"),
    ]
    print("Message Types Output:")
    for msg in messages:
        print(type(msg).__name__, "-", msg.content)
    print()


def demo_few_shot_prompting():
    """Demonstrates FewShotChatMessagePromptTemplate for providing examples to the LLM."""
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    fewshot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )

    print("Few-Shot Final Prompt Configuration:")
    print(final_prompt.format_messages(input="happy"))

    # model = init_chat_model(model="gpt-4o-mini", temperature=0)
    # response = model.invoke(final_prompt.format_messages(input="happy"))
    # print(response.content)
    print()


def demo_reusable_components():
    """Demonstrates combining prompt templates."""
    system_prompt = ChatPromptTemplate.from_messages([("system", "You are a {role}.")])
    user_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

    # Combine prompts
    full_prompt = system_prompt + user_prompt

    fin = full_prompt.format_messages(role="helpful assistant", question="What is AI?")
    print("Combined Prompts Output:")
    print(fin)
    print()


if __name__ == "__main__":
    demo_chat_prompt_template()
    demo_multi_message_templates()
    demo_message_types()
    demo_few_shot_prompting()
    demo_reusable_components()
