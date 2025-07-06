import json
import os
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from app.models.document import DocumentMetadata
from app.core.config import get_settings
from app.services.s3 import S3Service
from app.services.rag_service import RAGService
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.settings = get_settings()
        self.s3_service = S3Service()
        self.rag_service = RAGService()
        self.index_file = os.path.join(self.settings.faiss_index_dir, "documents_index.json")
        self._ensure_index_file()
    
    def _ensure_index_file(self):
        """인덱스 파일이 없으면 생성"""
        if not os.path.exists(self.index_file):
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump({"documents": {}}, f, ensure_ascii=False, indent=2)
    
    def save_document_metadata(self, document: DocumentMetadata) -> None:
        """문서 메타데이터를 JSON 파일에 저장"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 문서 정보 저장
            data["documents"][document.id] = document.dict()
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"문서 메타데이터 저장 실패: {e}")
            raise
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentMetadata]:
        """특정 문서 정보 조회"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if document_id in data["documents"]:
                doc_data = data["documents"][document_id]
                return DocumentMetadata(**doc_data)
            return None
            
        except Exception as e:
            logger.error(f"문서 조회 실패: {e}")
            return None
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        """모든 문서 목록 조회"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for doc_data in data["documents"].values():
                documents.append(DocumentMetadata(**doc_data))
            
            return sorted(documents, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"문서 목록 조회 실패: {e}")
            return []
    
    def update_document_status(self, document_id: str, status: str, **kwargs) -> None:
        """문서 상태 업데이트"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if document_id in data["documents"]:
                data["documents"][document_id]["status"] = status
                data["documents"][document_id]["updated_at"] = datetime.now().isoformat()
                
                # 추가 정보 업데이트
                for key, value in kwargs.items():
                    data["documents"][document_id][key] = value
                
                with open(self.index_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            logger.error(f"문서 상태 업데이트 실패: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """문서 삭제 (메타데이터 + FAISS 인덱스)"""
        try:
            # 메타데이터에서 삭제
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if document_id not in data["documents"]:
                return False
            
            # S3 키 가져오기
            s3_key = data["documents"][document_id].get("s3_key")
            
            # 메타데이터에서 제거
            del data["documents"][document_id]
            
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # FAISS 인덱스 폴더 삭제
            index_folder = os.path.join(self.settings.faiss_index_dir, document_id)
            if os.path.exists(index_folder):
                shutil.rmtree(index_folder)
            
            # S3 파일 삭제
            if s3_key:
                asyncio.create_task(self.s3_service.delete_file(s3_key))
            
            return True
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    def search_documents(self, query: str, file_types: Optional[List[str]] = None) -> List[DocumentMetadata]:
        """문서 검색"""
        try:
            documents = self.get_all_documents()
            
            # 검색 필터링
            filtered_docs = []
            for doc in documents:
                # 파일 타입 필터
                if file_types:
                    file_ext = Path(doc.filename).suffix.lower().lstrip('.')
                    if file_ext not in file_types:
                        continue
                
                # 텍스트 검색
                if query.lower() in doc.original_name.lower() or query.lower() in doc.filename.lower():
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    async def process_s3_folder(self, folder_path: str, file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """S3 폴더의 파일들을 일괄 처리"""
        try:
            # S3 폴더 스캔
            files = await self.s3_service.list_files_in_folder(folder_path)
            
            # 지원하는 파일 형식 필터링
            if file_extensions:
                supported_files = [f for f in files if any(f.lower().endswith(f'.{ext}') for ext in file_extensions)]
            else:
                supported_files = [f for f in files if any(f.lower().endswith(ext) for ext in ['.pdf', '.txt', '.docx', '.ppt', '.pptx'])]
            
            logger.info(f"처리할 파일 수: {len(supported_files)}")
            
            # 각 파일을 순차적으로 처리
            results = []
            for file_key in supported_files:
                try:
                    result = await self._process_single_file(file_key)
                    results.append(result)
                except Exception as e:
                    logger.error(f"파일 처리 실패 {file_key}: {e}")
                    results.append({
                        "file": file_key,
                        "status": "error",
                        "error": str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"S3 폴더 처리 실패: {e}")
            raise
    
    async def _process_single_file(self, s3_key: str) -> Dict[str, Any]:
        """단일 파일 처리"""
        try:
            # 파일 정보 추출
            filename = os.path.basename(s3_key)
            file_size = await self.s3_service.get_file_size(s3_key)
            content_type = self._get_content_type(filename)
            
            # 문서 메타데이터 생성
            document = DocumentMetadata(
                original_name=filename,
                filename=filename,
                file_size=file_size,
                content_type=content_type,
                s3_key=s3_key,
                status="processing"
            )
            
            # 메타데이터 저장
            self.save_document_metadata(document)
            
            # 파일 다운로드
            file_content = await self.s3_service.download_file(filename)
            
            # 임시 파일 저장
            temp_path = os.path.join(self.settings.upload_dir, filename)
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # RAG 처리
            start_time = datetime.now()
            result = self.rag_service.process_document(temp_path, document.id)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            if result["status"] == "success":
                # 상태 업데이트
                self.update_document_status(
                    document.id,
                    "completed",
                    chunk_count=result["chunk_count"],
                    processing_time=processing_time
                )
                
                return {
                    "file": filename,
                    "document_id": document.id,
                    "status": "success",
                    "chunk_count": result["chunk_count"],
                    "processing_time": processing_time
                }
            else:
                # 에러 상태 업데이트
                self.update_document_status(
                    document.id,
                    "error",
                    error_message=result.get("error"),
                    processing_time=processing_time
                )
                
                return {
                    "file": filename,
                    "document_id": document.id,
                    "status": "error",
                    "error": result.get("error"),
                    "processing_time": processing_time
                }
                
        except Exception as e:
            logger.error(f"파일 처리 실패 {s3_key}: {e}")
            raise
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _get_content_type(self, filename: str) -> str:
        """파일명으로 content-type 추정"""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        return content_types.get(ext, 'application/octet-stream') 