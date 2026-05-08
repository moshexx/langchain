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
    import subprocess

    print("--- Ollama Embeddings ---")

    # Helper: list local Ollama models via CLI
    def _list_ollama_models() -> list[str]:
        # Use plain text parsing since some Ollama versions don't support --json
        try:
            out = subprocess.check_output(["ollama", "list"]).decode("utf-8")
            models = []
            for line in out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
            return models
        except Exception:
            return []

    available = _list_ollama_models()
    if not available:
        print("No local Ollama models found. Either Ollama isn't running or no models are installed.")
        print("Options: 1) Install/pull an embedding model with `ollama pull <model>` 2) Run the HuggingFace or OpenAI demo instead.")
        print()
        return

    # Try to pick a model that looks like an embedding model
    preferred = None
    for name in available:
        lname = name.lower()
        if "embed" in lname or "embedding" in lname or "embedding" in name:
            preferred = name
            break

    # If not found, pick the first available model but warn it may not support embeddings
    model_to_use = preferred or available[0]
    if not preferred:
        print(f"No explicit embedding model found. Will try model: {model_to_use} (may not provide embeddings)")

    try:
        embeddings = OllamaEmbeddings(model=model_to_use)
        text = "This is a sample text."
        embedding = embeddings.embed_query(text)
        print(f"Length of Ollama embedding (model={model_to_use}): {len(embedding)}")
    except Exception as e:
        print(f"Could not run Ollama embeddings with model '{model_to_use}': {e}")
        # Try falling back to HuggingFace embeddings if available
        try:
            print("Falling back to HuggingFace embeddings demo...")
            demo_huggingface_embeddings()
            return
        except Exception as hf_e:
            print(f"HuggingFace demo failed or library not installed: {hf_e}")

        # If HuggingFace not available, try OpenAI embeddings if API key present
        if os.getenv("OPENAI_API_KEY"):
            try:
                print("Falling back to OpenAI embeddings demo...")
                demo_openai_embeddings()
                return
            except Exception as open_e:
                print(f"OpenAI demo failed: {open_e}")

        if not preferred:
            print("If this model doesn't support embeddings, pull a dedicated embedding model such as 'llama2-7b-embedding-q4_0' or install the HuggingFace/OpenAI dependencies to use those demos.")
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
    # demo_huggingface_embeddings()
    demo_ollama_embeddings() # Uncomment if you have Ollama running
    # demo_openai_embeddings()
