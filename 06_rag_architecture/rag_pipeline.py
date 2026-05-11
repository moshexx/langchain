"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

import tempfile
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language
models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI,
Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications.
Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability
platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


# ============================================================
# CORE LOGIC: Factory Functions & Classes
# ============================================================

def create_kb_vectorstore(text: str = KNOWLEDGE_BASE, source_name: str = "langchain_knowledge_base.md"):
    """Create a vector store from text content."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=text, metadata={"source": source_name}
    )
    chunks = splitter.split_documents([doc])

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=tempfile.mkdtemp(),
    )


def build_basic_rag_chain(vectorstore, model_name: str = DEFAULT_MODEL):
    """Builds a basic RAG chain."""
    llm = init_chat_model(model=model_name, temperature=0.2)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:

Make sure to answer in a concise manner, and if you don't know the answer, 
just say "I don't know." """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_with_sources_chain(vectorstore, model_name: str = DEFAULT_MODEL):
    """Builds a RAG chain that explicitly cites sources."""
    llm = init_chat_model(model=model_name, temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below.
        Include which sources you used.

        Context:
        {context}

        Question: {question}

        Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i + 1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    return (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_with_fallback_chain(vectorstore, model_name: str = DEFAULT_MODEL):
    """Builds a RAG chain with strict fallback instructions when data is missing."""
    llm = init_chat_model(model=model_name, temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based ONLY on the following context.
If the answer is not in the context, respond with:
"I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs_debug(docs):
        print(f"\n[Debug] Retrieved {len(docs)} documents from Vector DB:")
        for i, doc in enumerate(docs):
            print(f"  - Doc {i + 1}: {doc.page_content[:150]}...")
        print("[Debug] End of context\n")
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs_debug, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


class RAGResponse(BaseModel):
    """Structured schema for RAG response."""
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low")
    sources_used: List[str] = Field(description="List of sources referenced")
    follow_up: str = Field(description="Suggested follow-up question")


class DocumentQA:
    """Object-Oriented Document Q&A System with structured output."""
    def __init__(self, document: str, source_name: str = "document", model_name: str = DEFAULT_MODEL):
        self.vectorstore = create_kb_vectorstore(document, source_name)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.llm = init_chat_model(model=model_name, temperature=0.2)
        
        self.structured_llm = self.llm.with_structured_output(RAGResponse)

        self.prompt = ChatPromptTemplate.from_template(
            """
Based on the context below, answer the question. Rate your confidence.

Context:
{context}

Question: {question}

Provide a structured response."""
        )

    def _format_docs(self, docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in docs
        )

    def ask(self, question: str) -> RAGResponse:
        chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.structured_llm
        )
        return chain.invoke(question)


# ============================================================
# TEST / SIMULATION
# ============================================================

def demo_basic_rag():
    print("=" * 60)
    print("BASIC RAG DEMO")
    print("=" * 60)

    vector_store = create_kb_vectorstore()
    rag_chain = build_basic_rag_chain(vector_store)

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_sources():
    print("=" * 60)
    print("RAG WITH SOURCES")
    print("=" * 60)

    vectorstore = create_kb_vectorstore()
    rag_chain = build_rag_with_sources_chain(vectorstore)

    answer = rag_chain.invoke("What are the core components of LangChain?")
    print("Q: What are the core components?\n")
    print(f"A: {answer}")


def demo_rag_with_fallback():
    print("=" * 60)
    print("RAG WITH FALLBACK")
    print("=" * 60)

    vectorstore = create_kb_vectorstore()
    rag_chain = build_rag_with_fallback_chain(vectorstore)

    questions = [
        "What is the pricing for LangSmith?",  # In knowledge base
        "What is the stock price of OpenAI?",  # Not in knowledge base
        "How do I deploy LangChain to AWS?",  # Not in knowledge base
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_structured_rag():
    print("=" * 60)
    print("STRUCTURED RAG (Pydantic Output)")
    print("=" * 60)

    # Use the OOP DocumentQA class which supports structured output natively
    qa_system = DocumentQA(KNOWLEDGE_BASE, "langchain_knowledge_base.md")
    
    result = qa_system.ask("What is LangGraph?")

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")


def exercise_document_qa():
    print("=" * 60)
    print("EXERCISE: Document QA System")
    print("=" * 60)

    test_doc = """
    The Python programming language was created by Guido van Rossum.
    First released in 1991, Python emphasizes code readability.
    Python 3.12 was released in October 2023 with improved error messages.
    The language is named after Monty Python, not the snake.
    """

    qa = DocumentQA(test_doc, "python_facts")

    questions = [
        "Who created Python?",
        "When was Python 3.12 released?",
        "Why is Python named Python?",
    ]

    for q in questions:
        result = qa.ask(q)
        print(f"Q: {q}")
        print(f"A: {result.answer} [Confidence: {result.confidence}]\n")


if __name__ == "__main__":
    # demo_basic_rag()
    # demo_rag_with_sources()
    # demo_rag_with_fallback()
    # demo_structured_rag()
    exercise_document_qa()
