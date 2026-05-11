from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: Data Models & Factory Functions
# ============================================================

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")


class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


def build_string_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain that outputs a simple string."""
    prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")
    llm = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = StrOutputParser()
    return prompt | llm | parser


def build_json_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain that outputs a JSON object."""
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name' and 'age' for: {description}"
    )
    llm = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    parser = JsonOutputParser()
    return prompt | llm | parser


def build_pydantic_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain that outputs a Pydantic object using PydanticOutputParser."""
    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age', and 'occupation' for: {description}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    llm = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return prompt | llm | parser


def build_structured_movie_chain(model_name: str = DEFAULT_MODEL):
    """Builds a chain that uses .with_structured_output() for a MovieReview."""
    llm = init_chat_model(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return llm.with_structured_output(MovieReview)


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_string_demo():
    print("=" * 60)
    print("STRING OUTPUT PARSER DEMO")
    print("=" * 60)
    chain = build_string_chain()
    response = chain.invoke({"topic": "nature"})
    print(f"Type: {type(response)}")
    print(f"Response: {response}\n")


def run_json_demo():
    print("=" * 60)
    print("JSON OUTPUT PARSER DEMO")
    print("=" * 60)
    chain = build_json_chain()
    result = chain.invoke({"description": "A 25-year-old developer named Alex"})
    print(f"Type: {type(result)}")
    print(f"Result: {result}\n")


def run_pydantic_demo():
    print("=" * 60)
    print("PYDANTIC OUTPUT PARSER DEMO")
    print("=" * 60)
    chain = build_pydantic_chain()
    result = chain.invoke({"description": "A 30-year-old artist named Maria"})
    print(f"Type: {type(result)}")
    print(f"Result: {result}\n")


def run_structured_output_demo():
    print("=" * 60)
    print("STRUCTURED OUTPUT DEMO (.with_structured_output)")
    print("=" * 60)
    chain = build_structured_movie_chain()
    result = chain.invoke("Review: Inception is a mind-bending thriller. 9/10")
    print(f"Type: {type(result)}")
    print(f"Result: {result}\n")


if __name__ == "__main__":
    run_string_demo()
    run_json_demo()
    run_pydantic_demo()
    run_structured_output_demo()