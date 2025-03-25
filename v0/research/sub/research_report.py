import os
import getpass
from typing import TypedDict, Annotated, List
import operator
from v0.research.sub.research_analysts import Analyst
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langchain_core.messages import HumanMessage, SystemMessage

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

class ResearchPrompts:
    REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no preamble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## 주요 인사이트
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## 참고문헌` header.
8. List your sources in order and do not repeat.

IMPORTANT: You must write the entire report in Korean language, including:
- All section titles
- All content
- All summaries 
- Any other text content

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""



    INTRO_CONCLUSION_INSTRUCTIONS = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## 서론 as the section header. 

For your conclusion, use ## 결론 as the section header.

IMPORTANT: You must write the entire report in Korean language, including:
- All section titles 
- All content
- All summaries
- Any other text content

Here are the sections to reflect on for writing: {formatted_str_sections}"""


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


class ResearchGraph():
    def __init__(self, llm: ChatOpenAI):
        super().__init__()
        self.llm = llm
    
    def edge_initiate_all_interviews(self, state: ResearchGraphState):
        human_analyst_feedback = state.get("human_analyst_feedback", None)
        if human_analyst_feedback:
            return "create_analysts"
        else:
            topic = state["topic"]
            return [Send("conduct_interview", {"analyst": analyst,
                                               "topic": topic,
                                               "messages": [HumanMessage(
                                                   content=f"So you said you were writing an article on {topic}?"
                                                )]}) for analyst in state["analysts"]] 
        
    def node_write_introduction(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        
        instructions = ResearchPrompts.INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        intro = self.llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 

        return {"introduction": intro.content}
    

    def node_write_conclusion(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        
        instructions = ResearchPrompts.INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        conclusion = self.llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 

        return {"conclusion": conclusion.content}
    

    def node_write_report(self, state: ResearchGraphState):
        sections = state["sections"]
        topic = state["topic"]

        formatted_str_sections = "\n\n".join(f"{section}" for section in sections)

        system_message = ResearchPrompts.REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections)
        report = self.llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Write the report based up these memos.")])

        return {"content": report.content}
    

    def node_finalize_report(self, state: ResearchGraphState):
        """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
        # Save full final report
        content = state["content"]
        if content.startswith("## Insights"):
            content = content.strip("## Insights")
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        
        return {"final_report": final_report}