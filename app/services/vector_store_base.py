from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class VectorStoreMetrics:
    """벡터 스토어 작업 메트릭스 수집"""
    
    def __init__(self):
        self.operation_counts = {}
        self.operation_times = {}
        
    def track_operation(self, operation_name: str):
        """작업 추적을 위한 컨텍스트 매니저"""
        from contextlib import contextmanager
        import time
        
        @contextmanager
        def _tracker():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1
                self.operation_times[operation_name] = self.operation_times.get(operation_name, 0) + duration
        
        return _tracker()
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭스 정보 반환"""
        return {
            "operation_counts": self.operation_counts.copy(),
            "operation_times": self.operation_times.copy()
        }

class VectorStoreCache:
    """벡터 스토어 검색 결과 캐싱"""
    
    def __init__(self, max_size: int = 1000):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[List[Document]]:
        """캐시된 결과 조회"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: List[Document]) -> None:
        """결과 캐싱"""
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # 가장 오래된 항목 제거
        self.cache[key] = value

class VectorStoreBase(ABC):
    """벡터 스토어 기본 인터페이스"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """문서 추가"""
        pass
    
    @abstractmethod
    async def search_documents(self, query: str, k: int = 4, score_threshold: float = 0.3) -> List[Document]:
        """문서 검색"""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        pass
    
    @abstractmethod
    async def clear_all_documents(self) -> bool:
        """모든 문서 삭제"""
        pass
    
    @abstractmethod
    def get_vectorstore(self) -> FAISS:
        """벡터 저장소 인스턴스 반환"""
        pass 