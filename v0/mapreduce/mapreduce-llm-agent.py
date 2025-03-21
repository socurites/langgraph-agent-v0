import os
import getpass
from operator import add
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Annotated, TypedDict
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0) 


"""
Parallelization joke generation
- Take a user input {topic}
- Produce a list of {joke topics} from it
- Send each joke topic to joke generation code
"""

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int

class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": subject}) for subject in state["subjects"]]


# Joke generation
class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}

# Reduce: Best joke selection
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}

graph = StateGraph(OverallState)

graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)

graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, "generate_joke")
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

app =graph.compile()


for s in app.stream({"topic": "animals"}):
    print(s)