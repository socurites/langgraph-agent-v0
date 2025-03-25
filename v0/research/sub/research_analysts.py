from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import os
import getpass

"""
Setup
"""
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


"""
Generate Analysts: Human-In-The-Loop
"""
class ResearchPrompts:
    ANALYST_INSTRUCTIONS = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme.

6. IMPORTANT: All responses must be in Korean language. This includes:
   - Analyst names
   - Affiliations
   - Roles
   - Descriptions
   - Any other text content

7. The analysts should be relevant to the Korean market and context.""" 


class Analyst(BaseModel):
    affiliation: str = Field(
        description="The organization the analyst works for"
    )
    name: str = Field(
        description="The name of the analyst"
    )
    role: str = Field(
        description="The role of the analyst"
    )
    description: str = Field(
        description="Analyst's focus, concerns, and motives"
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nAffiliation: {self.affiliation}\nRole: {self.role}\nDescription: {self.description}"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts"
    )

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]

class AnalystCreationGraph():
    def __init__(self, llm: ChatOpenAI):
        super().__init__()
        self.llm = llm

    def node_create_analysts(self, state: GenerateAnalystsState):
        """ Create analysts """
        topic = state["topic"]
        max_analysts = state["max_analysts"]
        human_analyst_feedback = state.get("human_analyst_feedback", "")

        # Enforce structured output
        structured_llm = self.llm.with_structured_output(Perspectives)

        # System message
        system_message = ResearchPrompts.ANALYST_INSTRUCTIONS.format(topic=topic, max_analysts=max_analysts, human_analyst_feedback=human_analyst_feedback)
        response = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Geneerate the set of analysts")])

        return {"analysts": response.analysts}

    def node_human_feedback(self, state: GenerateAnalystsState):
        """ No-op node that should be interrupted by a human """
        pass

    def edge_should_continue(self, state: GenerateAnalystsState):
        human_analyst_feedback = state.get("human_analyst_feedback", None)
        if human_analyst_feedback:
            return "create_analysts"
        else:
            return END

def test_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    thisGraph = AnalystCreationGraph(llm)

    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", thisGraph.node_create_analysts)
    builder.add_node("human_feedback", thisGraph.node_human_feedback)

    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", thisGraph.edge_should_continue, ["create_analysts", END])

    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

    # Input
    input_max_analysts = 3
    input_topic = '''구글과 네이버 등, 검색 서비스가 AI 검색을 도입함에 따른 AISEO 또는 GEO 대응 전략'''
    input_human_analyst_feedback = "IT 서비스 기업가 관점을 추가하고 싶어. 스타트업에서 마케팅 전문가 출신의 사람도 추가해줘"
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream({"topic": input_topic, "max_analysts": input_max_analysts,}, thread, stream_mode="values"):
        analysts = event.get("analysts", '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("\n")

    # Human-in-the-loop
    graph.update_state(thread, {"human_analyst_feedback": input_human_analyst_feedback})

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get("analysts", '')
        if analysts:
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("\n")

    # If we are satisfied with the analysts, we can stop the graph execution
    further_feedback = None
    graph.update_state(thread, {"human_analyst_feedback": further_feedback}, as_node="human_feedback")

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)

    final_state = graph.get_state(thread)
    analysts = final_state.values.get("analysts", '')

    if analysts:
        print(f">>> Final Analysts:")
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")

if __name__ == "__main__":
    test_graph()