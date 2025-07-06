from typing import List, Tuple
from langchain.prompts import PromptTemplate

class PromptManager:
    """
    RAG 시스템에서 사용되는 프롬프트 템플릿을 관리하는 클래스
    """
    
    # 한국어 프롬프트 템플릿
    CONDENSE_QUESTION_TEMPLATE = """주어진 대화 기록을 참고하여 최신 질문에 대한 독립적인 질문을 만들어주세요.

대화 기록: {chat_history}
최신 질문: {question}

독립적인 질문:"""

    QA_TEMPLATE = """아래 제공된 컨텍스트를 사용하여 질문에 답변해주세요. 
컨텍스트에서 답을 찾을 수 없다면, "주어진 문서에서 해당 정보를 찾을 수 없습니다."라고 답변해주세요.
답변은 한국어로 해주세요.

컨텍스트: {context}

질문: {question}

답변:"""

    GENERAL_CHAT_TEMPLATE = """질문: {question}

답변을 한국어로 자연스럽게 작성해주세요."""

    CHAT_WITH_HISTORY_TEMPLATE = """이전 대화:
{chat_history}

현재 질문: {question}

위의 대화 맥락을 고려하여 현재 질문에 답변해주세요. 답변은 한국어로 작성해주세요."""

    @classmethod
    def get_condense_question_prompt(cls) -> PromptTemplate:
        """대화 기록을 고려한 질문 압축 프롬프트 반환"""
        return PromptTemplate(
            template=cls.CONDENSE_QUESTION_TEMPLATE,
            input_variables=["chat_history", "question"]
        )

    @classmethod
    def get_qa_prompt(cls) -> PromptTemplate:
        """문서 기반 질답 프롬프트 반환"""
        return PromptTemplate(
            template=cls.QA_TEMPLATE,
            input_variables=["context", "question"]
        )

    @classmethod
    def create_general_prompt(cls, question: str, chat_history: List[Tuple[str, str]] = None) -> str:
        """
        문서 없이 일반 질의를 위한 프롬프트 생성
        
        Args:
            question: 사용자 질문
            chat_history: 대화 기록 [(질문, 답변), ...]
        
        Returns:
            프롬프트 문자열
        """
        if not chat_history:
            return cls.GENERAL_CHAT_TEMPLATE.format(question=question)
        
        # 채팅 히스토리가 있는 경우 (최근 3개만 사용)
        history_text = ""
        for i, (q, a) in enumerate(chat_history[-3:]):
            history_text += f"이전 질문 {i+1}: {q}\n이전 답변 {i+1}: {a}\n\n"
        
        return cls.CHAT_WITH_HISTORY_TEMPLATE.format(
            chat_history=history_text,
            question=question
        ) 