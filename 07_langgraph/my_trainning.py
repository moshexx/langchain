from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

load_dotenv()
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


class QuizState(TypedDict):
    topic: str
    questions: str
    answers: str


def build_quiz_graph(model_name: str = DEFAULT_MODEL) -> CompiledStateGraph:
    llm = init_chat_model(model_name, temperature=DEFAULT_TEMPERATURE)

    def generate_questions_node(state: QuizState) -> dict:
        prompt = f"""You are a professor creating a multiple-choice quiz.
        Based on the following topic, generate 5 engaging questions.
        For each question, provide 4 answer options (one correct, three distractors).
        Return the questions in JSON format with the following structure:
        {{
            "questions": [
                {{
                    "question": "Question text?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct": "Correct option text"
                }}
            ]
        }}
        Topic: {state["topic"]}"""
        resp = llm.invoke(prompt)
        return {"questions": resp.content}

    def generate_answers_node(state: QuizState) -> dict:
        prompt = f"""You are a professor grading a multiple-choice quiz.
        Based on the following topic and questions, generate the correct answers.
        Return the answers in JSON format with the following structure:
        {{
            "answers": [
                {{
                    "question": "Question text?",
                    "correct": "Correct option text"
                }}
            ]
        }}
        Topic: {state["topic"]}
        Questions: {state["questions"]}"""
        resp = llm.invoke(prompt)
        return {"answers": resp.content}

    builder = StateGraph(QuizState)  # type: ignore
    builder.add_node("generate_questions", generate_questions_node)
    builder.add_node("generate_answers", generate_answers_node)
    builder.add_edge(START, "generate_questions")
    builder.add_edge("generate_questions", "generate_answers")
    builder.add_edge("generate_answers", END)
    return builder.compile()


def run_demo(topic: str):
    print(f"Running demo with topic: {topic}")
    app = build_quiz_graph()
    current_node = None

    for chunk, metadata in app.stream(
        {"topic": topic, "questions": "", "answers": ""},
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None  # type: ignore
        if node != current_node:
            if current_node is not None:
                print("\n")
            current_node = node
            label = "Questions" if node == "generate_questions" else "Answers"
            print(f"\n{'='*50}")
            print(f"  {label}")
            print(f"{'='*50}\n")

        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    run_demo("love")
