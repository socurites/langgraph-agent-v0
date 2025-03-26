"""
Research automation
-  lightweight, multi-agent system around chat models that customizes the research process

- Source Selection
  - Users can choose any set of input sources for their research
- Planning
  - Users provide a `topic`, and the system generates a team of AI analysts, each focusing on one sub-topic.
  - `Human-in-the-loop` will be used to refine these sub-topics before research begins.
- LLM Utilization
  - Each analyst will conduct in-depth interviews with an expert AI using the selected sources.
  - The interview will be a multi-turn conversation to extract detailed insights
  - These interviews will be captured in a using `sub-graphs` with their internal state
- Research Process
  - Experts will gather information to answer analyst questions in `parallel`
  - All interviews will be conducted simultaneously through `map-reduce`
- Output Format
  - The gathered insights from each interview will be synthesized into a final report
  - We'll use customizable prompts for the report, allowing for a flexible output format. 
"""

import os
import getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import List
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from v0.research.sub.research_analysts import AnalystCreationGraph
from v0.research.sub.research_interview import InterviewGraph
from v0.research.sub.research_report import ResearchGraph, ResearchGraphState

"""
Setup
"""
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


class ResearchAssistant:
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0

    def __init__(self):
        self.llm = ChatOpenAI(model=ResearchAssistant.LLM_MODEL, temperature=ResearchAssistant.LLM_TEMPERATURE)

        self.analyst_creation_graph = AnalystCreationGraph(self.llm)
        self.analyst_intervew_graph = InterviewGraph(self.llm)
        self.analyst_report_graph = ResearchGraph(self.llm)

        self._graph = self.build_graph()

    @property
    def graph(self):
        return self._graph

    def build_graph(self):
        builder = StateGraph(ResearchGraphState)
        builder.add_node("create_analysts", self.analyst_creation_graph.node_create_analysts)
        builder.add_node("human_feedback", self.analyst_creation_graph.node_human_feedback)
        builder.add_node("conduct_interview", self.analyst_intervew_graph.build_subgraph().compile())
        builder.add_node("write_report", self.analyst_report_graph.node_write_report)
        builder.add_node("write_introduction", self.analyst_report_graph.node_write_introduction)
        builder.add_node("write_conclusion", self.analyst_report_graph.node_write_conclusion)
        builder.add_node("finalize_report", self.analyst_report_graph.node_finalize_report)

        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.analyst_report_graph.edge_initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", END)
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)

        memory = MemorySaver()
        return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)    

def test_assistant(max_analysts, topic, max_num_turns):
    thread = {"configurable": {"thread_id": "1"}}

    assistant = ResearchAssistant()
    graph = assistant.graph

    for event in graph.stream({"topic": topic, "max_analysts": max_analysts,}, thread, stream_mode="values"):
        analysts = event.get("analysts", '')

    # Human-in-the-loop
    graph.update_state(thread, {"human_analyst_feedback": "IT 서비스 기업가 관점을 추가하고 싶어. 스타트업에서 마케팅 전문가 출신의 사람도 추가해줘"})

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get("analysts", '')

    # If we are satisfied
    further_feedback = None
    graph.update_state(thread, {"human_analyst_feedback": further_feedback}, as_node="human_feedback")

    # Continue the graph execution
    for event in graph.stream(None, thread, stream_mode="updates"):
        print("--Node--")
        node_name = next(iter(event.keys()))
        print(node_name)

    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    thread = {"configurable": {"thread_id": "1"}}
    interview = graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": max_num_turns}, thread)
    # print(interview['sections'][0])

    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    print(report)

# Export the graph for LangGraph platform
assistant = ResearchAssistant()
graph = assistant.graph

if __name__ == "__main__":
    # Input
    max_analysts = 5
    topic = '''구글과 네이버 등, 검색 서비스가 AI 검색을 도입함에 따른 AISEO 또는 GEO 대응 전략'''
    max_num_turns = 3

    test_assistant(max_analysts, topic, max_num_turns)