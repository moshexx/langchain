"""
Conversation Memory in LangChain
Modern approaches to maintaining conversation context
"""

import os
import sqlite3
from typing import Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
CHAT_HISTORY_DB_PATH = "./chat_history.db"
SQLITE_CONNECTION_STRING = f"sqlite:///{CHAT_HISTORY_DB_PATH}"


# ============================================================
# CORE LOGIC: Factory Functions & Classes
# (Copy these directly into your production application)
# ============================================================

def build_basic_memory_chain(model_name: str = DEFAULT_MODEL):
    """Builds a basic memory chain using InMemoryChatMessageHistory."""
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    store: Dict[str, InMemoryChatMessageHistory] = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
        
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    ), store


def build_windowed_memory_chain(model_name: str = DEFAULT_MODEL, window_size: int = 2):
    """Builds a memory chain that only keeps the last `window_size` exchanges."""
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    class WindowedChatHistory(InMemoryChatMessageHistory):
        k: int = window_size
        def add_messages(self, messages):
            super().add_messages(messages)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2):]

    store: Dict[str, WindowedChatHistory] = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(k=window_size)
        return store[session_id]
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    ), store


def build_persistent_memory_chain(model_name: str = DEFAULT_MODEL, connection_string: str = SQLITE_CONNECTION_STRING):
    """Builds a memory chain that persists to SQLite."""
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id, connection=connection_string
        )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember user preferences and facts."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


class SummaryMemorySystem:
    """End-to-end summary memory system using OOP: auto-summarize old messages, keep recent ones verbatim."""
    def __init__(self, model_name: str = DEFAULT_MODEL, max_recent: int = 4):
        self.summary_llm = ChatOpenAI(model=model_name, temperature=0)
        self.chat_llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.max_recent = max_recent
        
        self.running_summary = ""
        self.recent_messages = []
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Be concise.\n\nSummary of earlier conversation:\n{summary}"),
            MessagesPlaceholder(variable_name="recent_messages"),
            ("human", "{input}"),
        ])
        self.chat_chain = self.chat_prompt | self.chat_llm | StrOutputParser()
        
        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Condense the current summary and new messages into a single updated summary "
            "(2-3 sentences). Preserve all key facts about the user.\n\n"
            "Current summary:\n{current_summary}\n\n"
            "New messages:\n{new_messages}\n\n"
            "Updated summary:"
        )
        self.summarize_chain = self.summarize_prompt | self.summary_llm | StrOutputParser()

    def ask(self, user_input: str) -> str:
        response = self.chat_chain.invoke({
            "summary": self.running_summary if self.running_summary else "No prior conversation.",
            "recent_messages": self.recent_messages,
            "input": user_input,
        })
        
        self.recent_messages.append(HumanMessage(content=user_input))
        self.recent_messages.append(AIMessage(content=response))
        
        if len(self.recent_messages) > self.max_recent:
            messages_to_summarize = self.recent_messages[:-self.max_recent]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )
            self.running_summary = self.summarize_chain.invoke({
                "current_summary": self.running_summary if self.running_summary else "None yet.",
                "new_messages": formatted,
            })
            self.recent_messages = self.recent_messages[-self.max_recent:]
            
        return response


def trim_conversation_messages(messages, llm, max_tokens: int = 60):
    """Trims a list of messages to fit within a token limit."""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
    )


# ============================================================
# TEST / SIMULATION (How to run the logic)
# ============================================================

def demo_basic_memory():
    print("=" * 60)
    print("BASIC CONVERSATION MEMORY")
    print("=" * 60)

    # 1. Logic Initialization
    chain, store = build_basic_memory_chain()
    
    # 2. Config / Test Setup
    config = {"configurable": {"session_id": "user_123"}}
    messages = [
        "Hi! My name is Paulo.",
        "I'm learning about LangChain.",
        "What's my name and what am I learning?",
    ]

    # 3. Execution
    print("\nConversation:")
    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    print(f"\n--- Stored History ({len(store['user_123'].messages)} messages) ---")
    for msg in store["user_123"].messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:50]}...")


def demo_multi_sessions():
    print("=" * 60)
    print("MULTIPLE CONVERSATION SESSIONS")
    print("=" * 60)

    # 1. Logic Initialization
    chain, store = build_basic_memory_chain()
    
    # 2. Config / Test Setup
    user_a_config = {"configurable": {"session_id": "user_a"}}
    user_b_config = {"configurable": {"session_id": "user_b"}}

    # 3. Execution
    print("\n--- User A ---")
    print("User A: My favorite language is Python")
    resp = chain.invoke({"input": "My favorite language is Python"}, config=user_a_config)
    print(f"AI: {resp}")

    print("\n--- User B ---")
    print("User B: I love JavaScript")
    resp = chain.invoke({"input": "I love JavaScript"}, config=user_b_config)
    print(f"AI: {resp}")

    print("\n--- Asking each about their preference ---")
    for user_cfg, name in [(user_a_config, "User A"), (user_b_config, "User B")]:
        print(f"\n{name}: What's my favorite language?")
        resp = chain.invoke({"input": "What's my favorite language?"}, config=user_cfg)
        print(f"AI: {resp}")


def demo_message_trimming():
    print("=" * 60)
    print("MESSAGE TRIMMING")
    print("=" * 60)

    llm = init_chat_model(DEFAULT_MODEL)
    
    messages = [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a high-level programming language known for readability and versatility."),
        HumanMessage(content="How do I install it?"),
        AIMessage(content="You can install Python from python.org or use package managers."),
        HumanMessage(content="What about pip?"),
        AIMessage(content="Pip is Python's package installer. It comes with Python 3.4+."),
        HumanMessage(content="Can you summarize everything we discussed?"),
    ]

    print(f"\nOriginal: {len(messages)} messages")
    
    # 1. Logic Execution
    trimmed = trim_conversation_messages(messages, llm, max_tokens=60)

    print(f"After trimming (max 60 tokens): {len(trimmed)} messages")
    print("\nTrimmed messages:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"  {role}: {msg.content[:60]}...")


def demo_windowed_memory():
    print("=" * 60)
    print("WINDOWED MEMORY (Keep Last K)")
    print("=" * 60)

    # 1. Logic Initialization
    chain, store = build_windowed_memory_chain(window_size=2)
    
    # 2. Config / Test Setup
    config = {"configurable": {"session_id": "windowed_test"}}
    exchanges = [
        "My name is Paulo",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?",
    ]

    # 3. Execution
    print("\nConversation with k=2 window:")
    for msg in exchanges:
        print(f"\nUser: {msg}")
        response = chain.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

        history = store["windowed_test"].messages
        print(f"  [Window: {len(history)} msgs] ", end="")
        facts_in_memory = [m.content[:40] for m in history if isinstance(m, HumanMessage)]
        print(f"Remembers: {facts_in_memory}")

    print("\nRESULT: Window only kept last 2 exchanges!")


def demo_summary_memory():
    print("=" * 60)
    print("SUMMARY MEMORY")
    print("=" * 60)

    # 1. Logic Initialization
    system = SummaryMemorySystem(max_recent=4)
    
    # 2. Config / Test Setup
    exchanges = [
        "My name is Paulo and I'm from Seattle",
        "I work as an AI engineer building RAG systems",
        "I have 2 cats named Luna and Milo",
        "I'm building a LangChain course for Udemy",
        "What do you know about me? List everything.",
    ]

    # 3. Execution
    for user_input in exchanges:
        print(f"User: {user_input}")
        response = system.ask(user_input)
        print(f"AI: {response}\n")

    print("=" * 60)
    print("FINAL MEMORY STATE")
    print(f"\nRunning summary (compressed old context):\n  {system.running_summary}")
    print(f"\nRecent messages kept verbatim ({len(system.recent_messages)}):")
    for msg in system.recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}")


def exercise_persistent_memory_proof():
    print("=" * 60)
    print("EXERCISE: Persistent Memory Chatbot PROOF")
    print("=" * 60)

    # 1. Config / Test Setup
    if os.path.exists(CHAT_HISTORY_DB_PATH):
        os.remove(CHAT_HISTORY_DB_PATH)
        
    session_id = "persistent_user"
    config = {"configurable": {"session_id": session_id}}

    # RUN 1
    print("\n--- RUN 1: Storing preferences ---\n")
    chain_v1 = build_persistent_memory_chain()
    
    run1_messages = [
        "My name is Paulo. I prefer dark mode themes and Python over JavaScript.",
        "I also like my responses concise -- no fluff.",
    ]
    for msg in run1_messages:
        print(f"User: {msg}")
        response = chain_v1.invoke({"input": msg}, config=config)
        print(f"AI:   {response}\n")

    del chain_v1

    # RUN 2
    print("\n--- RUN 2: Fresh chain, testing recall ---\n")
    chain_v2 = build_persistent_memory_chain()
    
    recall_questions = [
        "What's my name?",
        "What programming language do I prefer?",
    ]
    for msg in recall_questions:
        print(f"User: {msg}")
        response = chain_v2.invoke({"input": msg}, config=config)
        print(f"AI:   {response}\n")

    del chain_v2

    # Verify DB
    print("--- FINAL DATABASE STATE ---\n")
    conn = sqlite3.connect(CHAT_HISTORY_DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM message_store")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"Total messages in DB after both runs: {count}")
    print("Everything was loaded from SQLite -- true persistence!")

    if os.path.exists(CHAT_HISTORY_DB_PATH):
        os.remove(CHAT_HISTORY_DB_PATH)


if __name__ == "__main__":
    # demo_basic_memory()
    # demo_multi_sessions()
    # demo_message_trimming()
    # demo_windowed_memory()
    # demo_summary_memory()
    exercise_persistent_memory_proof()
