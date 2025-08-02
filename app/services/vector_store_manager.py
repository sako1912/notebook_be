from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings.base import Embeddings
from pathlib import Path
import os
import logging
from app.services.vector_store_base import VectorStoreBase, VectorStoreMetrics

logger = logging.getLogger(__name__)

class VectorStoreManager(VectorStoreBase):
    """
    간단한 벡터 저장소 관리자
    - 하나의 통합된 벡터 DB
    - 초기화 시 기존 DB 로드 또는 신규 생성
    - 파일 업로드 시 벡터 DB에 추가
    - 질의 시 관련 문서 검색
    """
    
    def __init__(self, 
                embeddings: Embeddings, 
                persist_dir: str,
                text_splitter: TextSplitter):
        """
        Args:
            embeddings: 임베딩 모델
            persist_dir: 벡터 저장소 저장 경로
            text_splitter: 텍스트 분할기. None인 경우 기본값 사용
        """
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "faiss_index")
        
        # 텍스트 분할기 설정
        self.text_splitter = text_splitter
        
        # 메트릭스 초기화
        self.metrics = VectorStoreMetrics()
        
        # 저장 디렉토리 생성
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # 벡터 DB 초기화
        self.vectorstore = self._initialize_vectorstore()
        logger.info("벡터 저장소 초기화 완료")
    
    def _initialize_vectorstore(self) -> FAISS:
        """벡터 저장소 초기화"""
        with self.metrics.track_operation("initialize_vectorstore"):
            try:
                if os.path.exists(self.index_path):
                    vectorstore = FAISS.load_local(
                        self.index_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"기존 벡터 저장소 로드 완료: {self.index_path}")
                    return vectorstore
                else:
                    logger.info("기존 벡터 저장소가 없어 신규 생성합니다")
                    return self._create_empty_vectorstore()
                    
            except Exception as e:
                logger.error(f"벡터 저장소 로드 실패: {e}")
                logger.info("새로운 빈 벡터 저장소를 생성합니다")
                return self._create_empty_vectorstore()
    
    def _create_empty_vectorstore(self) -> FAISS:
        """빈 벡터 저장소 생성"""
        with self.metrics.track_operation("create_empty_vectorstore"):
            try:
                dummy_doc = Document(
                    page_content="초기화용 더미 문서입니다.",
                    metadata={"source": "system", "type": "dummy"}
                )
                
                vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
                vectorstore.save_local(self.index_path)
                
                logger.info("빈 벡터 저장소 생성 완료")
                return vectorstore
                
            except Exception as e:
                logger.error(f"빈 벡터 저장소 생성 실패: {e}")
                raise

    def get_vectorstore(self) -> FAISS:
        """벡터 저장소 인스턴스 반환"""
        return self.vectorstore

    async def add_documents(self, documents: List[Document], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """문서를 벡터 저장소에 추가"""
        with self.metrics.track_operation("add_documents"):
            try:
                # 문서를 청크로 분할
                chunks = self.text_splitter.split_documents(documents)
                
                if not chunks:
                    raise ValueError("문서에서 유효한 텍스트 청크를 생성할 수 없습니다")
                
                # 추가 메타데이터가 있으면 각 청크에 추가
                if metadata:
                    for chunk in chunks:
                        chunk.metadata.update(metadata)
                
                # 기존 벡터 저장소에 새 문서 추가
                new_vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)
                
                # 파일에 저장
                await self._save_vectorstore_async()
                
                logger.info(f"문서 {len(chunks)}개 청크가 벡터 저장소에 추가되었습니다")
                
                return {
                    "status": "success",
                    "chunk_count": len(chunks),
                    "message": f"{len(chunks)}개 청크가 성공적으로 추가되었습니다"
                }
                
            except Exception as e:
                logger.error(f"문서 추가 실패: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

    async def search_documents(self, query: str, k: int = 4, score_threshold: float = 0.3) -> List[Document]:
        """질의와 관련된 문서 검색"""
        with self.metrics.track_operation("search_documents"):
            try:
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={
                        "k": k,
                        "score_threshold": score_threshold
                    }
                )
                
                relevant_docs = await retriever.ainvoke(query)
                
                # 더미 문서 필터링
                filtered_docs = [
                    doc for doc in relevant_docs 
                    if doc.metadata.get("type") != "dummy"
                ]
                
                logger.info(f"검색 완료: {len(filtered_docs)}개 관련 문서 발견")
                return filtered_docs
                
            except Exception as e:
                logger.error(f"문서 검색 실패: {e}")
                return []

    async def _save_vectorstore_async(self) -> None:
        """벡터 저장소를 파일에 비동기로 저장"""
        with self.metrics.track_operation("save_vectorstore"):
            try:
                # 실제 저장은 동기 작업이지만, 비동기 컨텍스트에서 실행
                import asyncio
                await asyncio.to_thread(self.vectorstore.save_local, self.index_path)
                logger.debug(f"벡터 저장소 저장 완료: {self.index_path}")
            except Exception as e:
                logger.error(f"벡터 저장소 저장 실패: {e}")
                raise

    async def get_document_count(self) -> int:
        """저장된 문서 수 반환 (더미 문서 제외)"""
        with self.metrics.track_operation("get_document_count"):
            try:
                total_docs = self.vectorstore.index.ntotal
                return max(0, total_docs - 1)  # 더미 문서 1개 제외
            except Exception as e:
                logger.error(f"문서 수 조회 실패: {e}")
                return 0

    async def clear_all_documents(self) -> bool:
        """모든 문서 삭제 (더미 문서만 남김)"""
        with self.metrics.track_operation("clear_all_documents"):
            try:
                self.vectorstore = self._create_empty_vectorstore()
                logger.info("모든 문서가 삭제되었습니다")
                return True
            except Exception as e:
                logger.error(f"문서 삭제 실패: {e}")
                return False

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭스 정보 반환"""
        return self.metrics.get_metrics() 