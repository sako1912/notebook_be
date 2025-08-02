from typing import Dict, Any, List, Optional, Tuple
import logging

from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    RAG 오케스트레이터 - 마이크로서비스 조합
    
    모든 독립적인 서비스들을 조합하여 완전한 RAG 기능 제공:
    - DocumentService: 문서 처리
    - EmbeddingService: 임베딩 생성  
    - VectorStoreService: 벡터 저장
    - RetrievalService: 문서 검색
    - GenerationService: 답변 생성
    """
    
    def __init__(self):
        """RAG 오케스트레이터 초기화"""
        logger.info("RAG 오케스트레이터 초기화 시작...")
        
        # 각 마이크로서비스 초기화 (의존성 순서 고려)
        self.document_service = DocumentService()
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService(self.embedding_service)
        self.retrieval_service = RetrievalService(self.vector_store_service)
        self.generation_service = GenerationService()
        
        logger.info("RAG 오케스트레이터 초기화 완료")
    
    async def upload_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        파일 업로드 및 벡터 저장소에 저장 (전체 파이프라인)
        
        Args:
            file_path: 업로드할 파일 경로
            metadata: 문서 메타데이터
            
        Returns:
            처리 결과
        """
        try:
            # 1. 문서 처리 (DocumentService)
            doc_result = self.document_service.process_document(file_path, metadata)
            
            if doc_result["status"] == "error":
                return doc_result
            
            # 2. 벡터 저장소에 추가 (VectorStoreService)
            vector_result = await self.vector_store_service.add_documents(
                documents=doc_result["documents"],
                metadata=metadata
            )
            
            if vector_result["status"] == "success":
                # 결과 통합
                return {
                    "status": "success",
                    "file_path": file_path,
                    "document_count": doc_result["chunk_count"],
                    "total_documents_in_db": await self.vector_store_service.get_document_count(),
                    "message": f"파일 업로드 및 인덱싱 완료: {doc_result['chunk_count']}개 청크 생성"
                }
            else:
                return vector_result
                
        except Exception as e:
            error_msg = f"파일 업로드 파이프라인 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    def retriever(self, question: str, k: int = 4, search_type: str = "vector", 
                  document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        문서 검색 (기존 호환성 유지)
        
        Args:
            question: 검색 질의
            k: 반환할 문서 수
            search_type: 검색 유형
            document_id: 문서 ID (하위 호환성용)
            
        Returns:
            검색 결과
        """
        try:
            # RetrievalService를 통한 검색
            result = self.retrieval_service.search(
                query=question,
                k=k,
                search_type=search_type,
                include_scores=True
            )
            
            return result
            
        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "query": question
            }
    
    async def ask_question(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None,
                          search_type: str = "vector", max_docs: int = 4) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (RAG 방식)
        
        Args:
            question: 사용자 질문
            chat_history: 채팅 기록
            search_type: 검색 유형
            max_docs: 최대 문서 수
            
        Returns:
            답변 결과
        """
        try:
            if chat_history is None:
                chat_history = []
            
            # 1. 관련 문서 검색 (RetrievalService)
            search_result = self.retrieval_service.search(
                query=question,
                k=max_docs,
                search_type=search_type,
                include_scores=True
            )
            
            if search_result["status"] == "error":
                return {
                    "status": "error",
                    "error": f"문서 검색 실패: {search_result['error']}"
                }
            
            # 2. 답변 생성 (GenerationService)
            answer_result = await self.generation_service.generate_answer(
                question=question,
                context_documents=search_result.get("documents", []),
                chat_history=chat_history
            )
            
            if answer_result["status"] == "error":
                return answer_result
            
            # 3. 결과 통합
            return {
                "status": "success",
                "answer": answer_result["answer"],
                "source_documents": answer_result.get("source_documents", []),
                "relevant_chunks": answer_result.get("source_documents", []),
                "query_type": answer_result.get("generation_type", "rag"),
                "search_type": search_result.get("search_type", search_type),
                "context_used": answer_result.get("context_used", True)
            }
            
        except Exception as e:
            error_msg = f"질문 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "question": question
            }
    
    async def ask_question_conversational(self, question: str, 
                                        chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        대화형 RAG 질문 답변
        
        Args:
            question: 사용자 질문
            chat_history: 채팅 기록
            
        Returns:
            답변 결과
        """
        try:
            # 벡터 저장소 가져오기
            vectorstore = self.vector_store_service.get_vectorstore()
            
            # 대화형 답변 생성
            result = await self.generation_service.generate_conversational_answer(
                question=question,
                vectorstore=vectorstore,
                chat_history=chat_history
            )
            
            return result
            
        except Exception as e:
            error_msg = f"대화형 질문 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "question": question
            }
    
    async def summarize_documents(self, query: str = "", summary_type: str = "brief", 
                                max_docs: int = 10) -> Dict[str, Any]:
        """
        문서 요약 생성
        
        Args:
            query: 검색 쿼리 (빈 문자열이면 모든 문서)
            summary_type: 요약 타입
            max_docs: 최대 문서 수
            
        Returns:
            요약 결과
        """
        try:
            # 문서 검색
            if query:
                search_result = self.retrieval_service.search(query, k=max_docs)
                documents = search_result.get("documents", [])
            else:
                # 모든 문서 가져오기
                documents = self.vector_store_service.search_documents("", k=max_docs)
                documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            
            # 요약 생성
            summary_result = await self.generation_service.summarize_documents(
                documents=documents,
                summary_type=summary_type
            )
            
            return summary_result
            
        except Exception as e:
            error_msg = f"문서 요약 생성 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_database_status(self) -> Dict[str, Any]:
        """벡터 데이터베이스 상태 정보 반환"""
        try:
            return self.vector_store_service.get_status()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_database(self) -> Dict[str, Any]:
        """벡터 데이터베이스 초기화"""
        try:
            return self.vector_store_service.clear_all_documents()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return self.document_service.get_supported_extensions()
    
    def get_service_status(self) -> Dict[str, Any]:
        """모든 서비스 상태 정보 반환"""
        try:
            # 각 서비스 상태 확인
            embedding_cache_info = self.embedding_service.get_cache_info()
            vector_store_status = self.vector_store_service.get_status()
            search_stats = self.retrieval_service.get_search_stats()
            generation_stats = self.generation_service.get_generation_stats()
            
            return {
                "status": "success",
                "services": {
                    "document_service": {"status": "active", "supported_extensions": len(self.get_supported_extensions())},
                    "embedding_service": {
                        "status": "active",
                        "cache_size": embedding_cache_info.get("cache_size", 0),
                        "dimension": self.embedding_service.get_embedding_dimension()
                    },
                    "vector_store_service": {
                        "status": vector_store_status.get("status", "unknown"),
                        "document_count": vector_store_status.get("document_count", 0)
                    },
                    "retrieval_service": {
                        "status": search_stats.get("status", "unknown"),
                        "supported_search_types": search_stats.get("supported_search_types", [])
                    },
                    "generation_service": {
                        "status": generation_stats.get("status", "unknown"),
                        "llm_model": generation_stats.get("llm_model", "unknown")
                    }
                },
                "message": "RAG 오케스트레이터가 정상 작동 중입니다."
            }
            
        except Exception as e:
            error_msg = f"서비스 상태 조회 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    # 하위 호환성을 위한 별칭들
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """기존 호환성을 위한 메서드 (upload_document 호출)"""
        metadata = {"document_id": document_id} if document_id else None
        return self.upload_document(file_path, metadata)
    
    def get_document_list(self) -> List[str]:
        """저장된 문서 정보 반환 (호환성 유지)"""
        return []  # 간단한 구조에서는 문서ID 개념 없음 