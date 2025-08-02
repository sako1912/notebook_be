from typing import Dict, Any, List
from langchain_aws import BedrockEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from pathlib import Path
import os
import boto3
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingService:
    """
    임베딩 전용 서비스
    - 임베딩 모델 관리
    - 텍스트 벡터화
    - 임베딩 캐싱
    """
    
    def __init__(self):
        """임베딩 서비스 초기화"""
        logger.info("임베딩 서비스 초기화 시작...")
        
        # AWS Bedrock 클라이언트 초기화
        self.bedrock_client = self._initialize_bedrock_client()
        
        # 임베딩 모델 초기화
        self.embeddings = self._initialize_embeddings()
        
        logger.info("임베딩 서비스 초기화 완료")
    
    def _initialize_bedrock_client(self):
        """AWS Bedrock 클라이언트 초기화"""
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화 (캐시 포함)"""
        # 캐시 저장소 설정
        cache_dir = os.path.join(settings.base_dir, "embedding_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fs = LocalFileStore(cache_dir)
        
        # AWS Bedrock 임베딩 모델
        underlying_embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",  # 한국 리전에서 사용 가능
            region_name=settings.aws_region
        )
        
        # 캐시 지원 임베딩 생성
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            fs,
            namespace="bedrock_embeddings_cache"
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        try:
            embeddings = self.embeddings.embed_query(text)
            logger.debug(f"텍스트 임베딩 생성 완료: {len(text)} chars -> {len(embeddings)} dimensions")
            return embeddings
            
        except Exception as e:
            logger.error(f"텍스트 임베딩 생성 실패: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        다중 텍스트 임베딩 생성 (배치)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"배치 임베딩 생성 완료: {len(texts)} texts -> {len(embeddings)} vectors")
            return embeddings
            
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        임베딩 차원 수 반환
        
        Returns:
            임베딩 벡터 차원 수
        """
        try:
            # 샘플 텍스트로 차원 확인
            sample_embedding = self.embed_text("sample text")
            return len(sample_embedding)
            
        except Exception as e:
            logger.error(f"임베딩 차원 확인 실패: {str(e)}")
            # Titan 임베딩 모델의 기본 차원
            return 1536
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        임베딩 캐시 초기화
        
        Returns:
            초기화 결과
        """
        try:
            cache_dir = os.path.join(settings.base_dir, "embedding_cache")
            
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                
                logger.info("임베딩 캐시 초기화 완료")
                return {
                    "status": "success",
                    "message": "임베딩 캐시가 초기화되었습니다."
                }
            else:
                return {
                    "status": "info",
                    "message": "초기화할 캐시가 없습니다."
                }
                
        except Exception as e:
            error_msg = f"임베딩 캐시 초기화 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        임베딩 캐시 정보 반환
        
        Returns:
            캐시 정보
        """
        try:
            cache_dir = os.path.join(settings.base_dir, "embedding_cache")
            
            if not os.path.exists(cache_dir):
                return {
                    "cache_exists": False,
                    "cache_size": 0,
                    "file_count": 0
                }
            
            # 캐시 디렉토리 크기 계산
            total_size = 0
            file_count = 0
            
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
            
            return {
                "cache_exists": True,
                "cache_size": total_size,
                "file_count": file_count,
                "cache_path": cache_dir
            }
            
        except Exception as e:
            logger.error(f"캐시 정보 조회 실패: {str(e)}")
            return {
                "cache_exists": False,
                "error": str(e)
            } 