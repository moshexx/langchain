"""
Embeddings in LangChain

Model Dimensions & Costs:
- text-embedding-3-small	1536	$0.02 / 1M tokens	(General use)
- text-embedding-3-large	3072	$0.13 / 1M tokens	(High accuracy)
- text-embedding-ada-002	1536	$0.10 / 1M tokens	(Legacy)
"""

from dotenv import load_dotenv
import os

load_dotenv()

def demo_huggingface_embeddings() -> None:
    """Demonstrate embeddings using HuggingFace models."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    print("--- HuggingFace Embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions
    
    text = "This is a sample text to be embedded."
    embedding = embeddings.embed_query(text)
    print(f"Length of HF embedding: {len(embedding)}") # Should print 384
    print()


def demo_ollama_embeddings() -> None:
    """Demonstrate embeddings using local Ollama models."""
    from langchain_ollama import OllamaEmbeddings
    
    print("--- Ollama Embeddings ---")
    try:
        embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0")
        text = "This is a sample text."
        embedding = embeddings.embed_query(text)
        print(f"Length of Ollama embedding: {len(embedding)}")
    except Exception as e:
        print(f"Could not run Ollama (is it running locally?): {e}")
    print()


def demo_openai_embeddings() -> None:
    """Demonstrate embeddings using OpenAI models."""
    from langchain_openai.embeddings import OpenAIEmbeddings
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Skipping OpenAI embeddings demo.")
        return
        
    print("--- OpenAI Embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Single text
    text = "This is a sample text to be embedded."
    embedding = embeddings.embed_query(text)
    print(f"Length of single OpenAI embedding: {len(embedding)}")  # Should print 1536
    
    # Multiple texts
    embeds = embeddings.embed_documents(
        ["This is the first document.", "This is the second document."]
    )
    print(f"Number of embeddings returned for multiple texts: {len(embeds)}")  # Should print 2
    print(f"Length of each embedding: {len(embeds[0])}")  # Should print 1536
    print()


if __name__ == "__main__":
    demo_huggingface_embeddings()
    # demo_ollama_embeddings() # Uncomment if you have Ollama running
    demo_openai_embeddings()
