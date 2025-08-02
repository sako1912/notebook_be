import json
import os
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from app.models.document import DocumentMetadata
from app.core.config import get_settings
from app.services.s3 import S3Service
import asyncio
import logging
from datetime import datetime

from app.services.document_loader import DocumentLoader

logger = logging.getLogger(__name__)

class DocumentService:
    """
    문서 처리 전용 서비스
    - 파일 업로드 검증
    - 문서 로드 및 청킹
    - 메타데이터 생성
    """
    
    def __init__(self):
        """문서 서비스 초기화"""
        logger.info("문서 서비스 초기화 시작...")
        logger.info("문서 서비스 초기화 완료")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        파일 유효성 검사
        
        Args:
            file_path: 검사할 파일 경로
            
        Returns:
            검사 결과
        """
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": f"파일이 존재하지 않습니다: {file_path}"
                }
            
            # 지원하는 파일 형식인지 확인
            if not DocumentLoader.is_supported(file_path):
                return {
                    "status": "error",
                    "error": f"지원하지 않는 파일 형식입니다: {Path(file_path).suffix}"
                }
            
            return {
                "status": "success",
                "message": "파일 유효성 검사 통과"
            }
            
        except Exception as e:
            error_msg = f"파일 유효성 검사 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def process_document(self, file_path: str, additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        문서 처리 (로드 및 메타데이터 생성)
        
        Args:
            file_path: 처리할 파일 경로
            additional_metadata: 추가 메타데이터
            
        Returns:
            처리된 문서와 메타데이터
        """
        try:
            # 파일 유효성 검사
            validation_result = self.validate_file(file_path)
            if validation_result["status"] == "error":
                return validation_result
            
            # 문서 로드
            documents = DocumentLoader.load_document(file_path)
            
            if not documents:
                return {
                    "status": "error",
                    "error": "문서에서 텍스트를 추출할 수 없습니다"
                }
            
            # 파일 정보를 메타데이터에 추가
            file_metadata = self._generate_metadata(file_path, additional_metadata)
            
            # 각 문서 청크에 메타데이터 추가
            for doc in documents:
                doc.metadata.update(file_metadata)
            
            logger.info(f"문서 처리 완료: {file_path}, 청크 수: {len(documents)}")
            
            return {
                "status": "success",
                "documents": documents,
                "metadata": file_metadata,
                "chunk_count": len(documents)
            }
            
        except Exception as e:
            error_msg = f"문서 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    def _generate_metadata(self, file_path: str, additional_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        파일 메타데이터 생성
        
        Args:
            file_path: 파일 경로
            additional_metadata: 추가 메타데이터
            
        Returns:
            생성된 메타데이터
        """
        file_metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": Path(file_path).suffix,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
        if additional_metadata:
            file_metadata.update(additional_metadata)
        
        return file_metadata
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return list(DocumentLoader.SUPPORTED_EXTENSIONS) 