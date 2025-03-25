import os
import getpass
from typing import Annotated
import operator
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from v0.research.sub.research_analysts import Analyst
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string, AIMessage
from langgraph.checkpoint.memory import MemorySaver

"""
Setup
"""
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var} environment variable: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


class ResearchPrompts:
    QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


    SEARCH_INSTRUCTIONS = """You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""


    ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""


    SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

IMPORTANT: You must write the entire report in Korean language, including:
- All section titles
- All content
- All summaries
- All source descriptions
- Any other text content

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
- Verify that all content is in Korean language"""




class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="The search query to use for the search engine")

class InterviewGraph:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # Web search
        self.tavily_search = TavilySearchResults(max_results=3)

    def node_generate_question(self, state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]

        system_message = ResearchPrompts.QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
        question = self.llm.invoke([SystemMessage(content=system_message)] + messages)

        return {"messages": [question]}
    
    def node_search_web(self, state: InterviewState):
        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SystemMessage(content=ResearchPrompts.SEARCH_INSTRUCTIONS)] + state["messages"])

        # Search
        search_docs = self.tavily_search.invoke(search_query.search_query)

        formatted_docs = "\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_docs]}
    
    def node_search_wikipedia(self, state: InterviewState):
        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SystemMessage(content=ResearchPrompts.SEARCH_INSTRUCTIONS)] + state["messages"])

        # Search
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]} 
    
    def node_generate_answer(self, state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]

        system_message = ResearchPrompts.ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
        answer = self.llm.invoke([SystemMessage(content=system_message)] + messages)

        answer.name = "expert"

        return {"messages": [answer]}
    
    def node_save_interview(self, state: InterviewState):
        messages = state["messages"]

        interview = get_buffer_string(messages)

        return {"interview": interview}
    
    def node_write_section(self, state: InterviewState):
        # Get state
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
    
        # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
        system_message = ResearchPrompts.SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
        section = self.llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this interview and sources to write your section:\n\nInterview:\n{interview}\n\nSources:\n{context}")]) 
                    
        # Append it to state
        return {"sections": [section.content]}

    def edge_route_messages(self, state: InterviewState, name: str = "expert"):
        messages = state["messages"]
        max_num_turns = state.get("max_num_turns", 2)

        # Check the number of expert answers
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )

        if num_responses >= max_num_turns:
            return 'save_interview'
        
        last_question = messages[-2]

        if "Thank you so much for your help" in last_question.content:
            return "save_interview"
        return "ask_question"
    
    def build_subgraph(self):
        thisGraph = InterviewGraph(self.llm)

        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", thisGraph.node_generate_question)
        interview_builder.add_node("search_web", thisGraph.node_search_web)
        interview_builder.add_node("search_wikipedia", thisGraph.node_search_wikipedia)
        interview_builder.add_node("answer_question", thisGraph.node_generate_answer)
        interview_builder.add_node("save_interview", thisGraph.node_save_interview)
        interview_builder.add_node("write_section", thisGraph.node_write_section)


        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", thisGraph.edge_route_messages,['ask_question','save_interview'])
        interview_builder.add_edge("save_interview", "write_section")

        return interview_builder


def test_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    thisGraph = InterviewGraph(llm).build_subgraph()

    memory = MemorySaver()
    interview_graph = thisGraph.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")
   
if __name__ == "__main__":
    test_graph()