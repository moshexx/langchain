# LangChain Learning Project

A hands-on learning project exploring LangChain with OpenAI and Anthropic models.

## Topics Covered

- **Core Concepts** (`core_concepts.py`) — LangChain fundamentals
- **Working with LLMs** (`working_with_llms.py`) — Interacting with language models
- **Prompt Templates** (`prompt_templates_all.py`) — Building and using prompt templates
- **Prompt Messages** (`prompt_messages.py`) — Chat message formatting
- **Output Parsers** (`output_parsers_demo.py`, `output_parsers_final.py`) — Parsing LLM outputs
- **Smart Bot** (`smart_bot_section1.py`) — Building a conversational bot
- **Backend API** (`backend.py`) — FastAPI backend integration

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repo
git clone https://github.com/moshecohen/langchain.git
cd langchain

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env
```

### Environment Variables

Edit `.env` with your API keys:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LANGCHAIN_API_KEY=...
```

## Running Examples

```bash
uv run python core_concepts.py
uv run python working_with_llms.py
uv run python prompt_templates_all.py
uv run python output_parsers_demo.py
```

### Backend API

```bash
uv run uvicorn backend:app --reload
```
