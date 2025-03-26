"""
LangGraph 배포를 위한 환경 설정
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeploymentConfig:
    # LLM 설정
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0
    
    # API 키 설정
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # 로깅 설정
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 그래프 설정
    max_analysts: int = 5
    max_num_turns: int = 3
    
    @classmethod
    def validate(cls):
        """환경 설정 유효성 검증"""
        config = cls()
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not config.tavily_api_key:
            raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")
        return config 