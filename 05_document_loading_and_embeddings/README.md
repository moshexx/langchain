# Document Loading & Embeddings

This folder covers the data ingestion and retrieval components of LangChain.

## Files
- **`document_loaders.py`**: Shows how to load documents from various sources like text files, PDFs, and web pages.
- **`text_splitters.py`**: Demonstrates strategies for chunking text for Vector databases (recursive character, markdown header, code splitters).
- **`embeddings.py`**: Examples of using different embedding models (HuggingFace, Ollama, OpenAI) to convert text into vector representations.
- **`vector_stores.py`**: Shows how to store embeddings in a vector database (Chroma) and how to use it as a retriever for similarity search.
