"""
Output Parsers and Structured Output in LangChain V.1
"""

from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

load_dotenv()

# ============================================================
# GLOBAL CONSTANTS
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


# ============================================================
# CORE LOGIC: Data Models & Factory Functions
# ============================================================

class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    ingredients: List[str] = Field(description="List of ingredients")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    difficulty: str = Field(description="easy, medium, or hard")


class TaskExtraction(BaseModel):
    """Extracted task information."""
    task: str = Field(description="The main task to do")
    priority: str = Field(description="high, medium, or low")
    deadline: Optional[str] = Field(description="Deadline if mentioned")
    assignee: Optional[str] = Field(description="Person assigned if mentioned")


class Address(BaseModel):
    street: str
    city: str
    country: str


class Company(BaseModel):
    name: str
    industry: str
    employee_count: int
    headquarters: Address
    products: List[str]


class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Year released")
    director: str = Field(description="Director name")
    actors: List[str] = Field(description="Main actors")
    genre: str = Field(description="Primary genre")
    rating: int = Field(description="Rating from 1-10", ge=1, le=10)


def build_str_chain(model_name: str = DEFAULT_MODEL):
    """Builds a basic string output chain."""
    prompt = ChatPromptTemplate.from_template("Give me a one-word answer: What color is the sky?")
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return prompt | model | StrOutputParser()


def build_json_chain(model_name: str = DEFAULT_MODEL):
    """Builds a JSON output chain."""
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return prompt | model | JsonOutputParser()


def build_pydantic_recipe_chain(model_name: str = DEFAULT_MODEL):
    """Builds a Pydantic output chain for recipes."""
    parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    return prompt | model | parser


def build_structured_task_chain(model_name: str = DEFAULT_MODEL):
    """Builds a structured output chain for tasks."""
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    structured_model = model.with_structured_output(TaskExtraction)
    prompt = ChatPromptTemplate.from_template("Extract task information from: {text}")
    return prompt | structured_model


def build_structured_company_chain(model_name: str = DEFAULT_MODEL):
    """Builds a structured output chain for companies."""
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    structured_model = model.with_structured_output(Company)
    prompt = ChatPromptTemplate.from_template("Extract company information from: {text}")
    return prompt | structured_model


def build_structured_movie_chain(model_name: str = DEFAULT_MODEL):
    """Builds a structured output chain for movies."""
    model = ChatOpenAI(model=model_name, temperature=DEFAULT_TEMPERATURE)
    structured_model = model.with_structured_output(Movie)
    prompt = ChatPromptTemplate.from_template("Extract movie information from this review:\n\n{review}")
    return prompt | structured_model


# ============================================================
# TEST / SIMULATION
# ============================================================

def run_str_demo():
    print("=" * 60)
    print("STRING PARSER DEMO")
    print("=" * 60)
    chain = build_str_chain()
    result = chain.invoke({})
    print(f"Result: '{result}'\n")


def run_json_demo():
    print("=" * 60)
    print("JSON PARSER DEMO")
    print("=" * 60)
    chain = build_json_chain()
    result = chain.invoke({"place": "The Eiffel Tower"})
    print(f"Result: {result}\n")


def run_pydantic_demo():
    print("=" * 60)
    print("PYDANTIC PARSER DEMO")
    print("=" * 60)
    chain = build_pydantic_recipe_chain()
    result = chain.invoke({"dish": "scrambled eggs"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins\n")


def run_task_demo():
    print("=" * 60)
    print("STRUCTURED TASK EXTRACTION DEMO")
    print("=" * 60)
    chain = build_structured_task_chain()
    texts = [
        "John needs to finish the report by Friday - it's urgent",
        "Critical: Fix the login bug ASAP",
    ]
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"Input: {text}")
        print(f"  Task: {result.task}, Priority: {result.priority}\n")


def run_company_demo():
    print("=" * 60)
    print("STRUCTURED COMPANY EXTRACTION DEMO")
    print("=" * 60)
    chain = build_structured_company_chain()
    result = chain.invoke(
        {
            "text": "Apple Inc. is a tech company based in Cupertino, California, USA. They make iPhones."
        }
    )
    print(f"Company: {result.name}, Industry: {result.industry}")
    print(f"HQ: {result.headquarters.city}, {result.headquarters.country}\n")


def run_movie_exercise():
    print("=" * 60)
    print("EXERCISE: MOVIE EXTRACTION")
    print("=" * 60)
    chain = build_structured_movie_chain()
    result = chain.invoke(
        {
            "review": "The Dark Knight (2008) directed by Christopher Nolan is a masterpiece. 10/10!"
        }
    )
    print(f"Title: {result.title}, Year: {result.year}, Rating: {result.rating}/10\n")


if __name__ == "__main__":
    run_str_demo()
    run_json_demo()
    run_pydantic_demo()
    run_task_demo()
    run_company_demo()
    run_movie_exercise()
