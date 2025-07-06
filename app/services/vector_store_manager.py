from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from pathlib import Path
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    벡터 저장소 관리를 담당하는 클래스
    """
    
    def __init__(self, embeddings: Embeddings, persist_dir: str):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 저장 디렉토리 생성
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document], document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        문서로부터 벡터 저장소를 생성하고 저장
        
        Args:
            documents: 처리할 문서 리스트
            document_id: 문서 ID (제공되면 별도 인덱스 생성)
        
        Returns:
            처리 결과 정보
        """
        try:
            # 문서를 청크로 분할
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("문서에서 유효한 텍스트 청크를 생성할 수 없습니다.")
            
            # 벡터 저장소 생성
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # 저장 경로 결정
            if document_id:
                save_path = self._get_document_index_path(document_id)
            else:
                save_path = self._get_main_index_path()
            
            # 인덱스 저장
            self._save_vectorstore(vectorstore, save_path)
            
            # 청크 정보 계산
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            
            logger.info(f"벡터 저장소 생성 완료: {save_path}, 청크 수: {len(chunks)}")
            
            return {
                "status": "success",
                "chunk_count": len(chunks),
                "average_chunk_size": int(avg_size),
                "save_path": save_path
            }
            
        except Exception as e:
            logger.error(f"벡터 저장소 생성 실패: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def load_vectorstore(self, document_id: Optional[str] = None) -> Optional[FAISS]:
        """
        벡터 저장소 로드
        
        Args:
            document_id: 문서 ID (제공되면 해당 문서 인덱스 로드)
        
        Returns:
            FAISS 벡터 저장소 또는 None
        """
        try:
            if document_id:
                load_dir = os.path.dirname(self._get_document_index_path(document_id))
            else:
                load_dir = os.path.dirname(self._get_main_index_path())
            
            # 먼저 FAISS 내장 방식으로 로드 시도
            try:
                vectorstore = FAISS.load_local(load_dir, self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"벡터 저장소 로드 완료: {load_dir}")
                return vectorstore
            except Exception as e:
                logger.warning(f"FAISS 내장 로드 실패: {e}")
                
                # 대안: pickle 파일로 로드
                if document_id:
                    load_path = self._get_document_index_path(document_id)
                else:
                    load_path = self._get_main_index_path()
                
                if not os.path.exists(load_path):
                    logger.warning(f"벡터 저장소 파일이 존재하지 않습니다: {load_path}")
                    return None
                
                with open(load_path, "rb") as f:
                    vectorstore = pickle.load(f)
                
                # 임베딩 함수 복원
                if hasattr(vectorstore, 'embedding_function'):
                    vectorstore.embedding_function = self.embeddings
                
                logger.info(f"벡터 저장소 pickle 로드 완료: {load_path}")
                return vectorstore
            
        except Exception as e:
            logger.error(f"벡터 저장소 로드 실패: {str(e)}")
            return None
    
    def delete_vectorstore(self, document_id: str) -> bool:
        """
        특정 문서의 벡터 저장소 삭제
        
        Args:
            document_id: 문서 ID
        
        Returns:
            삭제 성공 여부
        """
        try:
            doc_dir = os.path.join(self.persist_dir, document_id)
            if os.path.exists(doc_dir):
                import shutil
                shutil.rmtree(doc_dir)
                logger.info(f"벡터 저장소 삭제 완료: {doc_dir}")
                return True
            else:
                logger.warning(f"삭제할 벡터 저장소가 존재하지 않습니다: {doc_dir}")
                return False
                
        except Exception as e:
            logger.error(f"벡터 저장소 삭제 실패: {str(e)}")
            return False
    
    def list_document_ids(self) -> List[str]:
        """
        저장된 문서 ID 목록 반환
        
        Returns:
            문서 ID 리스트
        """
        try:
            document_ids = []
            for item in os.listdir(self.persist_dir):
                item_path = os.path.join(self.persist_dir, item)
                if os.path.isdir(item_path) and item != "__pycache__":
                    index_path = os.path.join(item_path, "faiss_index.pkl")
                    if os.path.exists(index_path):
                        document_ids.append(item)
            
            return document_ids
            
        except Exception as e:
            logger.error(f"문서 ID 목록 조회 실패: {str(e)}")
            return []
    
    def _get_main_index_path(self) -> str:
        """메인 인덱스 파일 경로 반환"""
        return os.path.join(self.persist_dir, "faiss_index.pkl")
    
    def _get_document_index_path(self, document_id: str) -> str:
        """문서별 인덱스 파일 경로 반환"""
        doc_dir = os.path.join(self.persist_dir, document_id)
        Path(doc_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(doc_dir, "faiss_index.pkl")
    
    def _save_vectorstore(self, vectorstore: FAISS, save_path: str) -> None:
        """벡터 저장소를 파일로 저장"""
        try:
            # FAISS 내장 저장 방식 사용
            save_dir = os.path.dirname(save_path)
            vectorstore.save_local(save_dir)
            logger.debug(f"벡터 저장소 저장 완료: {save_dir}")
        except Exception as e:
            logger.error(f"벡터 저장소 저장 실패: {e}")
            # 대안: pickle 사용 (임베딩 클라이언트 제거)
            try:
                import copy
                vectorstore_copy = copy.deepcopy(vectorstore)
                # 임베딩 함수에서 클라이언트 제거
                if hasattr(vectorstore_copy, 'embedding_function'):
                    if hasattr(vectorstore_copy.embedding_function, 'client'):
                        vectorstore_copy.embedding_function.client = None
                
                with open(save_path, "wb") as f:
                    pickle.dump(vectorstore_copy, f)
                logger.debug(f"벡터 저장소 pickle 저장 완료: {save_path}")
            except Exception as e2:
                logger.error(f"벡터 저장소 pickle 저장도 실패: {e2}")
                raise 