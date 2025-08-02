from typing import Dict, Any, List, Optional, Tuple
from langchain.chains import ConversationalRetrievalChain
import logging

from app.services.llm_service import LLMService
from app.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class GenerationService:
    """
    답변 생성 전용 서비스
    - LLM 기반 답변 생성
    - 프롬프트 관리
    - 컨텍스트 처리
    """
    
    def __init__(self):
        """답변 생성 서비스 초기화"""
        logger.info("답변 생성 서비스 초기화 시작...")
        
        # LLM 서비스 초기화
        self.llm_service = LLMService()
        
        logger.info("답변 생성 서비스 초기화 완료")
    
    async def generate_answer(self, question: str, context_documents: List[Dict], 
                            chat_history: Optional[List[Tuple[str, str]]] = None,
                            max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        컨텍스트 기반 답변 생성
        
        Args:
            question: 사용자 질문
            context_documents: 검색된 컨텍스트 문서들
            chat_history: 채팅 기록
            max_tokens: 최대 토큰 수
            
        Returns:
            생성된 답변
        """
        try:
            if chat_history is None:
                chat_history = []
            
            # 컨텍스트가 없으면 일반 답변
            if not context_documents:
                return await self._generate_fallback_answer(question)
            
            # 컨텍스트 문서들을 텍스트로 변환
            context_text = self._format_context_documents(context_documents)
            
            # 프롬프트 생성
            prompt = self._create_rag_prompt(question, context_text, chat_history)
            
            # LLM으로 답변 생성
            answer = await self.llm_service.generate_async(
                prompt=prompt,
                max_tokens=max_tokens
            )
            
            logger.info(f"RAG 답변 생성 완료: {question[:50]}...")
            
            return {
                "status": "success",
                "answer": answer,
                "source_documents": context_documents,
                "context_used": True,
                "generation_type": "rag"
            }
            
        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "question": question
            }
    
    async def generate_conversational_answer(self, question: str, vectorstore,
                                           chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        대화형 RAG 답변 생성 (ConversationalRetrievalChain 사용)
        
        Args:
            question: 사용자 질문
            vectorstore: 벡터 저장소
            chat_history: 채팅 기록
            
        Returns:
            생성된 답변
        """
        try:
            if chat_history is None:
                chat_history = []
            
            # QA 체인 생성
            qa_chain = self._create_qa_chain(vectorstore)
            
            # 답변 생성
            response = await qa_chain.ainvoke({
                "question": question,
                "chat_history": chat_history
            })
            
            logger.info(f"대화형 RAG 답변 생성 완료: {question[:50]}...")
            
            return {
                "status": "success",
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "context_used": True,
                "generation_type": "conversational_rag"
            }
            
        except Exception as e:
            error_msg = f"대화형 답변 생성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "question": question
            }
    
    async def _generate_fallback_answer(self, question: str) -> Dict[str, Any]:
        """
        컨텍스트가 없을 때의 대체 답변
        
        Args:
            question: 사용자 질문
            
        Returns:
            대체 답변
        """
        try:
            fallback_answer = (
                "죄송합니다. 업로드된 문서에서 관련 정보를 찾을 수 없습니다. "
                "다른 질문을 해주시거나 관련 문서를 업로드해 주세요."
            )
            
            return {
                "status": "success",
                "answer": fallback_answer,
                "source_documents": [],
                "context_used": False,
                "generation_type": "fallback"
            }
            
        except Exception as e:
            logger.error(f"대체 답변 생성 실패: {str(e)}")
            raise
    
    def _format_context_documents(self, documents: List[Dict]) -> str:
        """
        컨텍스트 문서들을 텍스트로 포맷팅
        
        Args:
            documents: 문서 리스트
            
        Returns:
            포맷팅된 컨텍스트 텍스트
        """
        try:
            context_parts = []
            
            for i, doc in enumerate(documents, 1):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # 소스 정보 추가
                source_info = ""
                if "file_name" in metadata:
                    source_info = f" (출처: {metadata['file_name']})"
                
                context_parts.append(f"문서 {i}{source_info}:\n{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"컨텍스트 포맷팅 실패: {str(e)}")
            return ""
    
    def _create_rag_prompt(self, question: str, context: str, 
                          chat_history: List[Tuple[str, str]]) -> str:
        """
        RAG용 프롬프트 생성
        
        Args:
            question: 질문
            context: 컨텍스트
            chat_history: 채팅 기록
            
        Returns:
            생성된 프롬프트
        """
        try:
            # 채팅 기록 포맷팅
            history_text = ""
            if chat_history:
                history_parts = []
                for human_msg, ai_msg in chat_history[-3:]:  # 최근 3개만 사용
                    history_parts.append(f"사용자: {human_msg}")
                    history_parts.append(f"어시스턴트: {ai_msg}")
                history_text = "\n".join(history_parts) + "\n\n"
            
            # 프롬프트 템플릿
            prompt_template = """다음 문서들을 참고하여 질문에 답변해주세요.

제공된 문서:
{context}

{history}현재 질문: {question}

답변 시 다음 규칙을 따라주세요:
1. 제공된 문서의 내용을 바탕으로만 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 가능한 한 구체적이고 정확한 답변을 제공하세요
4. 출처를 명시하여 답변하세요
5. 문서에서 답을 찾을 수 없다면 솔직히 모른다고 하세요

답변:"""
            
            return prompt_template.format(
                context=context,
                history=history_text,
                question=question
            )
            
        except Exception as e:
            logger.error(f"프롬프트 생성 실패: {str(e)}")
            return f"질문: {question}\n답변:"
    
    def _create_qa_chain(self, vectorstore):
        """RAG를 위한 QA 체인 생성"""
        try:
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm_service.llm,
                retriever=vectorstore.as_retriever(
                    search_kwargs={
                        "k": 4,
                        "score_threshold": 0.3
                    }
                ),
                condense_question_prompt=PromptManager.get_condense_question_prompt(),
                combine_docs_chain_kwargs={"prompt": PromptManager.get_qa_prompt()},
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            logger.error(f"QA 체인 생성 실패: {str(e)}")
            raise
    
    async def summarize_documents(self, documents: List[Dict], 
                                 summary_type: str = "brief") -> Dict[str, Any]:
        """
        문서 요약 생성
        
        Args:
            documents: 요약할 문서들
            summary_type: 요약 타입 ("brief", "detailed", "bullet_points")
            
        Returns:
            생성된 요약
        """
        try:
            if not documents:
                return {
                    "status": "error",
                    "error": "요약할 문서가 없습니다."
                }
            
            # 문서 내용 합치기
            combined_content = self._format_context_documents(documents)
            
            # 요약 프롬프트 생성
            summary_prompt = self._create_summary_prompt(combined_content, summary_type)
            
            # 요약 생성
            summary = await self.llm_service.generate_async(summary_prompt)
            
            logger.info(f"문서 요약 생성 완료: {len(documents)}개 문서")
            
            return {
                "status": "success",
                "summary": summary,
                "summary_type": summary_type,
                "document_count": len(documents)
            }
            
        except Exception as e:
            error_msg = f"문서 요약 생성 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def _create_summary_prompt(self, content: str, summary_type: str) -> str:
        """
        요약용 프롬프트 생성
        
        Args:
            content: 요약할 내용
            summary_type: 요약 타입
            
        Returns:
            요약 프롬프트
        """
        if summary_type == "brief":
            instruction = "다음 문서들의 내용을 간략하게 요약해주세요."
        elif summary_type == "detailed":
            instruction = "다음 문서들의 내용을 자세하게 요약해주세요. 주요 포인트들을 모두 포함하세요."
        elif summary_type == "bullet_points":
            instruction = "다음 문서들의 주요 내용을 불릿 포인트 형태로 정리해주세요."
        else:
            instruction = "다음 문서들의 내용을 요약해주세요."
        
        return f"""{instruction}

문서 내용:
{content}

요약:"""
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        생성 서비스 통계 정보
        
        Returns:
            통계 정보
        """
        try:
            llm_info = self.llm_service.get_model_info()
            
            return {
                "status": "success",
                "llm_model": llm_info.get("model_name", "Unknown"),
                "supported_generation_types": ["rag", "conversational_rag", "fallback", "summary"],
                "max_context_length": llm_info.get("max_tokens", "Unknown"),
                "message": "답변 생성 서비스가 정상 작동 중입니다."
            }
            
        except Exception as e:
            error_msg = f"생성 서비스 통계 조회 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            } 