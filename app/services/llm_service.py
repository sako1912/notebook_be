from typing import Dict, Any, List, Tuple, Optional
from langchain_aws import ChatBedrock
from langchain.schema.messages import HumanMessage
import boto3
import logging
from app.core.config import get_settings
from app.core.llm_config import LLMConfig, LLMModelConfig
from app.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """
    LLM (Large Language Model) 서비스
    
    AI 모델과의 직접적인 상호작용을 담당
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """LLM 서비스 초기화"""
        # 모델명이 지정되지 않았으면 설정에서 가져오기
        if model_name is None:
            model_name = settings.llm_model
            
        self.model_config = LLMConfig.get_model_config(model_name)
        self.bedrock_client = self._initialize_bedrock_client()
        self.llm = self._initialize_llm()
        logger.info(f"LLM 서비스 초기화 완료 - 모델: {self.model_config.model_id}")
    
    def _initialize_bedrock_client(self):
        """AWS Bedrock 클라이언트 초기화"""
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=self.model_config.region
        )
    
    def _initialize_llm(self):
        """LLM 모델 초기화"""
        return ChatBedrock(
            client=self.bedrock_client,
            model_id=self.model_config.model_id,
            model_kwargs=self.model_config.to_model_kwargs()
        )
    
    async def generate(
        self, 
        question: str, 
        context: str = "", 
        chat_history: List[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        질문 처리 (컨텍스트 포함)
        
        Args:
            question: 사용자 질문
            context: 참조할 컨텍스트 (빈 문자열이면 일반 대화)
            chat_history: 대화 기록
        
        Returns:
            LLM 응답 결과
        """
        try:
            chat_history = chat_history or []
            
            # 컨텍스트가 포함된 프롬프트 생성
            if context.strip():
                # 컨텍스트가 있는 경우
                prompt = PromptManager.create_context_prompt(question, context, chat_history)
                query_type = "context_based"
            else:
                # 컨텍스트가 없는 경우 (일반 대화)
                prompt = PromptManager.create_general_prompt(question, chat_history)
                query_type = "general"
            
            # LLM 실행
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            logger.info(f"LLM 질의 처리 완료 ({query_type}): {question[:50]}...")
            
            return {
                "status": "success",
                "answer": response.content,
                "query_type": query_type
            }
            
        except Exception as e:
            error_msg = f"LLM 질의 처리 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """현재 사용 중인 모델 정보 반환"""
        return self.model_config.to_info_dict()
    
    def switch_model(self, model_name: str) -> bool:
        """모델 변경"""
        try:
            new_config = LLMConfig.get_model_config(model_name)
            self.model_config = new_config
            self.bedrock_client = self._initialize_bedrock_client()
            self.llm = self._initialize_llm()
            logger.info(f"모델 변경 완료: {self.model_config.model_id}")
            return True
        except Exception as e:
            logger.error(f"모델 변경 실패: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return LLMConfig.list_available_models() 