from typing import Dict, Any, List, Optional, Tuple
import logging
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.llm_service import LLMService
from app.services.vector_store_manager import VectorStoreManager
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QueryService:
    """
    질의 처리 통합 서비스
    
    RAG 검색과 LLM 응답을 조율하는 중앙 서비스
    """
    
    def __init__(self):
        """QueryService 초기화"""
        self.rag_service = RAGOrchestrator()  # RAGService → RAGOrchestrator로 변경
        self.llm_service = LLMService()
        logger.info("QueryService 초기화 완료")
    
    async def process_query(
        self,
        question: str,
        document_id: Optional[str] = None,
        chat_history: List[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        질의 처리 메인 로직
        
        Args:
            question: 사용자 질문
            document_id: 특정 문서 ID (선택사항)
            chat_history: 대화 기록 [(질문, 답변), ...]
        
        Returns:
            질의 처리 결과
        """
        try:
            chat_history = chat_history or []
            
            # 1. 문서 검색 (항상 수행)
            search_result = self.rag_service.retriever(
                question=question,
                document_id=document_id,
                search_type="hybrid"  # 하이브리드 검색 사용
            )
            documents = search_result.get("documents", [])
            
            # 2. 문서들을 컨텍스트로 변환 (없으면 빈 문자열)
            context = self._documents_to_context(documents)
            
            # 3. LLM 호출 
            response = await self.llm_service.generate(
                question=question,
                context=context,
                chat_history=chat_history
            )
            
            # 4. 응답 형태 통일
            if response["status"] == "success":
                return {
                    "status": "success",
                    "answer": response["answer"],
                    "source_documents": documents,
                    "relevant_chunks": [
                        {
                            "content": doc.get("content", str(doc)) if isinstance(doc, dict) else (doc.page_content if hasattr(doc, 'page_content') else str(doc)),
                            "metadata": doc.get("metadata", {}) if isinstance(doc, dict) else (doc.metadata if hasattr(doc, 'metadata') else {}),
                            "score": doc.get("score", None) if isinstance(doc, dict) else getattr(doc, 'score', None)
                        }
                        for doc in documents
                    ],
                    "query_type": "rag" if documents else "general"
                }
            else:
                return response
                
        except Exception as e:
            error_msg = f"질의 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def _documents_to_context(self, documents: List[Any]) -> str:
        """
        검색된 문서들을 컨텍스트 문자열로 변환
        
        Args:
            documents: 검색된 문서들
            
        Returns:
            컨텍스트 문자열
        """
        if not documents:
            return ""
            
        context_parts = []
        
        for i, doc in enumerate(documents[:4]):  # 상위 4개만 사용
            if isinstance(doc, dict):
                content = doc.get("content", str(doc))
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
                
            context_parts.append(f"[문서 {i+1}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 반환"""
        return {
            "service": "QueryService",
            "description": "질의 처리 통합 서비스",
            "components": {
                "rag_service": "RAG 오케스트레이터 (문서 검색 및 처리)",
                "llm_service": "AI 응답 생성 담당"
            },
            "llm_model": self.llm_service.get_model_info()
        } 