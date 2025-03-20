"""
Wikipedia와 Web-Search를 사용하여 주어진 질문에 대한 답변을 생성하는 Agent
"""
from langchain_openai import ChatOpenAI
from typing import Annotated
import operator
from typing_extensions import TypedDict
import os
import getpass
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


llm = ChatOpenAI(model="gpt-4o", temperature=0)

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]

def search_web(state):
    """Search the web for information"""

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state["question"])

    # Format
    formatted_docs = "\n\n---\n\n".join(
        [
            f'<Docment href="{doc["url"]}/>\n{doc["content"]}' for doc in search_docs
        ]
    )

    # Return
    return {"context": [formatted_docs]}

def search_wikipedia(state):
    """Search Wikipedia for information"""

    # Search
    search_docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()

    # Format
    formatted_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page")}"/>\n{doc.page_content}' for doc in search_docs
        ]
    )

    # Return
    return {"context": [formatted_docs]}

def generate_answer(state):
    """Generate an answer to the question"""

    context = state["context"]
    question = state["question"]

    answer_template = """Answer the question {question} using this context: {context} in Korean"""
    answer_instructions = answer_template.format(question=question, context=context)

    answer = llm.invoke([SystemMessage(content=answer_instructions)] + [HumanMessage(content="Answer the question")])

    return {"answer": answer}


builder = StateGraph(State)

builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "search_web")
builder.add_edge(START, "search_wikipedia")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

result = graph.invoke({"question": "AI Agent 시대에 커머스 기업의 대응 전략은?"})
print(result["answer"].content)