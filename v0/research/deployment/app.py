"""
LangGraph 플랫폼과 통합된 Research Assistant API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from langchain_core.messages import HumanMessage
from v0.research.research_assistant import ResearchAssistant
from v0.research.deployment.config import DeploymentConfig

# LangGraph 설정 로드
with open(os.path.join(os.path.dirname(__file__), "langgraph.json")) as f:
    langgraph_config = json.load(f)

app = FastAPI(title="Research Assistant API")
config = DeploymentConfig.validate()

# 연구 요청 모델
class ResearchRequest(BaseModel):
    topic: str
    max_analysts: Optional[int] = langgraph_config["config"]["max_analysts"]
    max_num_turns: Optional[int] = langgraph_config["config"]["max_num_turns"]
    analyst_feedback: Optional[str] = None

@app.post("/research")
async def create_research(request: ResearchRequest) -> Dict[str, Any]:
    """새로운 연구 프로젝트 시작"""
    try:
        # LangGraph 설정으로 ResearchAssistant 초기화
        assistant = ResearchAssistant()
        graph = assistant.graph
        
        # 초기 상태 설정
        thread = {"configurable": {"thread_id": "1"}}
        initial_state = {
            "topic": request.topic,
            "max_analysts": request.max_analysts,
            "max_num_turns": request.max_num_turns
        }
        
        # 그래프 실행
        for event in graph.stream(initial_state, thread, stream_mode="values"):
            analysts = event.get("analysts", '')

        # Human-in-the-loop
        # 실제로는 다시 입력 받아야 함
        graph.update_state(thread, {"human_analyst_feedback": "IT 서비스 기업가 관점을 추가하고 싶어. 스타트업에서 마케팅 전문가 출신의 사람도 추가해줘"})

        # Continue the graph execution
        for event in graph.stream(None, thread, stream_mode="values"):
            analysts = event.get("analysts", '')

        further_feedback = None
        graph.update_state(thread, {"human_analyst_feedback": further_feedback}, as_node="human_feedback")

        # Continue the graph execution
        for event in graph.stream(None, thread, stream_mode="updates"):
            print("--Node--")
            node_name = next(iter(event.keys()))
            print(node_name)

        messages = [HumanMessage(f"So you said you were writing an article on {initial_state.get('topic')}?")]
        interview = graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": initial_state.get('max_num_turns')}, thread)    

        # 최종 결과 생성
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        
        return {
            "status": "success",
            "report": report,
            "analysts": analysts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "config": langgraph_config
    } 