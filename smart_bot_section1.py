"""
Section 1 Project: Smart Q&A Bot
A production-ready question-answering bot with structured output.
Refactored for clarity, modularity, and clean code principles.
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable, Client

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# --- 1. Configuration & Setup ---

def setup_environment():
    """Initialize environment variables and LangSmith tracing."""
    load_dotenv()
    
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "Smart Q&A Bot Project")
        print(f"🚀 LangSmith configured - Project: {os.environ.get('LANGCHAIN_PROJECT')}")
    else:
        print("⚠️ LangSmith API Key not found. Tracing is disabled.")


# --- 2. Schema Definitions ---

class QAResponse(BaseModel):
    """Structured output format for the bot's responses."""
    answer: str = Field(description="The final answer to the user's question.")
    confidence: str = Field(description="Confidence level: high, medium, or low.")
    reasoning: str = Field(description="The step-by-step reasoning behind the answer.")
    follow_up_questions: List[str] = Field(
        description="A list of relevant follow-up questions.",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Whether external sources/search would improve the answer.",
        default=False,
    )


# --- 3. Core Bot Implementation ---

class SmartQABot:
    """A bot that provides structured, traceable, and reliable answers."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(QAResponse)
        
        self.prompt = self._create_prompt_template()
        self.chain = self.prompt | self.model

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Define the system instructions and message structure."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a highly capable and honest Q&A assistant.
            
Your guidelines:
- Provide accurate and concise answers.
- Be transparent about uncertainty; set confidence to 'low' if you aren't sure.
- Explain your reasoning clearly.
- Suggest 2-3 logical follow-up questions.
- Flag if external information (sources) would be beneficial."""),
            ("human", "{question}"),
        ])

    @traceable(name="ask_single_question")
    def ask(self, question: str) -> QAResponse:
        """Process a single question and return a structured response."""
        try:
            return self.chain.invoke({"question": question})
        except Exception as e:
            return self._get_error_response(e)

    @traceable(name="ask_multiple_questions")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """Process multiple questions in parallel using LangChain's batching."""
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)

    def _get_error_response(self, error: Exception) -> QAResponse:
        """Generate a fallback response when the LLM call fails."""
        return QAResponse(
            answer="I encountered an error while processing your request.",
            confidence="low",
            reasoning=f"Error details: {str(error)}",
            follow_up_questions=["Would you like to try rephrasing the question?"],
            sources_needed=True
        )


# --- 4. UI & Formatting Helpers ---

def print_separator(char: str = "=", length: int = 60):
    print(char * length)

def print_header(title: str):
    print_separator()
    print(title.center(60))
    print_separator()

def display_response(question: str, response: QAResponse):
    """Format and print the bot's structured output."""
    print(f"\n❓ QUESTION: {question}")
    print(f"🤖 ANSWER:   {response.answer}")
    print(f"📊 CONFIDENCE: {response.confidence.upper()}")
    print(f"🧠 REASONING: {response.reasoning}")
    print(f"🔍 FOLLOW-UP: {', '.join(response.follow_up_questions)}")
    if response.sources_needed:
        print("ℹ️  Note: This answer might benefit from external source verification.")
    print("-" * 60)


# --- 5. Demo Scenarios ---

def run_standard_demo(bot: SmartQABot):
    """Run a basic Q&A demonstration."""
    print_header("STANDARD Q&A DEMO")
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
    ]
    for q in questions:
        res = bot.ask(q)
        display_response(q, res)

def run_batch_demo(bot: SmartQABot):
    """Run a batch processing demonstration."""
    print_header("BATCH PROCESSING DEMO")
    questions = [
        "What is Python?",
        "What is Rust?",
    ]
    responses = bot.ask_batch(questions)
    for q, r in zip(questions, responses):
        display_response(q, r)

def run_error_demo(bot: SmartQABot):
    """Run an error handling demonstration with a simulated issue."""
    print_header("ERROR HANDLING DEMO")
    # A very long input that might cause issues or just demonstrating the mechanism
    weird_input = "Tell me everything about " + "nothing " * 50
    res = bot.ask(weird_input)
    display_response(weird_input, res)


# --- 6. Main Execution ---

def main():
    """Main entry point to run all demos."""
    setup_environment()
    
    # Initialize the bot
    bot = SmartQABot()
    
    try:
        run_standard_demo(bot)
        run_batch_demo(bot)
        run_error_demo(bot)
        
        print_header("SECTION 1 COMPLETE")
        print("\nLearning Points:")
        print("✅ Modularized setup and configuration")
        print("✅ Structured output using Pydantic")
        print("✅ Organized class-based bot logic")
        print("✅ Clean formatting and UI helpers")
        print("✅ LangSmith tracing integration")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Optional: ensure traces are flushed if using high volume
        # Client().flush()
        pass

if __name__ == "__main__":
    main()
