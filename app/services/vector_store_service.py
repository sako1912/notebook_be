from typing import Dict, Any, List, Optional
import logging

from app.services.vector_store_manager import VectorStoreManager
from app.services.embedding_service import EmbeddingService
from app.core.config import get_settings
from app.models.document import DocumentMetadata
from app.services.vector_store_base import VectorStoreCache, VectorStoreMetrics
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorStoreService:
    """
    벡터 저장소 전용 서비스
    - 벡터 저장소 관리
    - 문서 추가/삭제
    - 벡터 검색
    - 메타데이터 저장
    """
    
    def __init__(self, 
                embedding_service: EmbeddingService,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                text_splitter: Optional[TextSplitter] = None):
        """
        벡터 저장소 서비스 초기화
        
        Args:
            embedding_service: 임베딩 서비스
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 중복 크기 (기본값: 200)
            text_splitter: 커스텀 텍스트 분할기 (기본값: None)
        """
        logger.info("벡터 저장소 서비스 초기화 시작...")
        
        # 임베딩 서비스 주입
        self.embedding_service = embedding_service
        
        # 메트릭스 초기화
        self.metrics = VectorStoreMetrics()
        
        # 캐시 초기화
        self.cache = VectorStoreCache(max_size=1000)
        
        # 텍스트 분할기 설정
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        
        # 벡터 저장소 초기화
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embedding_service.embeddings,
            persist_dir=settings.faiss_index_dir,
            text_splitter=text_splitter
        )
        
        logger.info("벡터 저장소 서비스 초기화 완료")
    
    async def add_documents(self, documents: List, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """문서들을 벡터 저장소에 추가"""
        with self.metrics.track_operation("add_documents"):
            try:
                result = await self.vector_store_manager.add_documents(documents, metadata)
                
                if result["status"] == "success":
                    logger.info(f"벡터 저장소에 문서 추가 완료: {result.get('chunk_count', 0)}개")
                
                return result
                
            except Exception as e:
                error_msg = f"문서 추가 실패: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg
                }
    
    async def search_documents(self, query: str, k: int = 4, search_type: str = "vector") -> Dict[str, Any]:
        """문서 검색 (캐시 지원)"""
        with self.metrics.track_operation("search_documents"):
            try:
                # 캐시 확인
                cache_key = f"{query}:{k}:{search_type}"
                cached_result = await self.cache.get(cache_key)
                
                if cached_result:
                    logger.info("캐시된 검색 결과 반환")
                    return {
                        "status": "success",
                        "documents": cached_result,
                        "source": "cache",
                        "search_type": search_type
                    }
                
                # 벡터 검색 수행
                documents = await self.vector_store_manager.search_documents(
                    query=query,
                    k=k,
                    score_threshold=0.3
                )
                
                # 결과 캐싱
                await self.cache.set(cache_key, documents)
                
                return {
                    "status": "success",
                    "documents": documents,
                    "source": "vectorstore",
                    "search_type": search_type
                }
                
            except Exception as e:
                error_msg = f"문서 검색 실패: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg
                }
    
    async def get_status(self) -> Dict[str, Any]:
        """벡터 저장소 상태 정보"""
        with self.metrics.track_operation("get_status"):
            try:
                doc_count = await self.vector_store_manager.get_document_count()
                metrics = self.vector_store_manager.get_metrics()
                
                return {
                    "status": "active",
                    "document_count": doc_count,
                    "metrics": metrics,
                    "embedding_dimension": self.embedding_service.get_embedding_dimension()
                }
                
            except Exception as e:
                error_msg = f"상태 조회 실패: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg
                }
    
    async def clear_all_documents(self) -> Dict[str, Any]:
        """모든 문서 삭제"""
        with self.metrics.track_operation("clear_all_documents"):
            try:
                success = await self.vector_store_manager.clear_all_documents()
                if success:
                    return {
                        "status": "success",
                        "message": "모든 문서가 삭제되었습니다"
                    }
                else:
                    return {
                        "status": "error",
                        "error": "문서 삭제 실패"
                    }
            except Exception as e:
                error_msg = f"문서 삭제 실패: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg
                }
    
    def get_vectorstore(self):
        """벡터 저장소 인스턴스 반환"""
        return self.vector_store_manager.get_vectorstore()
    
    def get_metrics(self) -> Dict[str, Any]:
        """서비스 메트릭스 정보 반환"""
        return self.metrics.get_metrics()
        
    async def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return await self.vector_store_manager.get_document_count() 