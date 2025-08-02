from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class LLMModelConfig:
    """LLM 모델 설정"""
    model_id: str
    provider: str
    temperature: float
    max_tokens: int
    region: str = "ap-northeast-2"
    
    def to_model_kwargs(self) -> Dict[str, Any]:
        """모델 초기화용 kwargs 반환"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    def to_info_dict(self) -> Dict[str, Any]:
        """모델 정보 딕셔너리 반환"""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "region": self.region
        }


class LLMConfig:
    """LLM 설정 관리 클래스"""
    
    # Claude 3.5 Sonnet 설정
    CLAUDE_3_5_SONNET = LLMModelConfig(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        provider="AWS Bedrock",
        temperature=0.1,
        max_tokens=4000,
        region="ap-northeast-2"
    )
    
    # Claude 3 Haiku 설정 (빠른 응답용)
    CLAUDE_3_HAIKU = LLMModelConfig(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        provider="AWS Bedrock",
        temperature=0.1,
        max_tokens=2000,
        region="ap-northeast-2"
    )
    
    # Titan Text 설정 (비용 효율적)
    TITAN_TEXT = LLMModelConfig(
        model_id="amazon.titan-text-express-v1",
        provider="AWS Bedrock",
        temperature=0.1,
        max_tokens=3000,
        region="ap-northeast-2"
    )
    
    # 기본 모델 설정
    DEFAULT_MODEL = CLAUDE_3_5_SONNET
    
    @classmethod
    def get_model_config(cls, model_name: str = "default") -> LLMModelConfig:
        """모델 설정 반환"""
        model_map = {
            "default": cls.DEFAULT_MODEL,
            "claude-3.5-sonnet": cls.CLAUDE_3_5_SONNET,
            "claude-3-haiku": cls.CLAUDE_3_HAIKU,
            "titan": cls.TITAN_TEXT
        }
        
        return model_map.get(model_name, cls.DEFAULT_MODEL)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return {
            "claude-3.5-sonnet": "Claude 3.5 Sonnet (고성능, 균형)",
            "claude-3-haiku": "Claude 3 Haiku (빠른 응답)",
            "titan": "Amazon Titan Text (비용 효율적)"
        } 