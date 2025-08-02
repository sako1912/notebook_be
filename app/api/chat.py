from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.query_service import QueryService
from app.services.rag_orchestrator import RAGOrchestrator  # RAGService → RAGOrchestrator
from app.services.s3 import S3Service
from app.services.document_service import DocumentService
from app.models.document import DocumentRequest
import os
import re
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

# 서비스 인스턴스들
query_service = QueryService()  # 질의 처리용
rag_service = RAGOrchestrator()  # RAGService → RAGOrchestrator
s3_service = S3Service()
document_service = DocumentService()

def clean_text(text: str) -> str:
    """텍스트에서 제어 문자와 문제가 될 수 있는 문자들을 제거합니다."""
    if not isinstance(text, str):
        return str(text)
    
    # 제어 문자 제거 (탭, 개행 문자는 유지)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # 연속된 공백을 하나로 줄이기
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

@router.post("/process/{filename}")
async def process_document(filename: str):
    """
    S3에서 파일을 다운로드하고 RAG 시스템에 처리합니다.
    """
    try:
        # S3에서 파일 다운로드
        file_content = await s3_service.download_file(filename)
        
        # 임시 파일로 저장
        temp_path = os.path.join(settings.upload_dir, filename)
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # RAG 처리 (upload_document 사용)
        result = rag_service.upload_document(temp_path)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"문서 처리 중 오류 발생: {result['error']}"
            )

        return {
            "message": "문서가 성공적으로 처리되었습니다.",
            "document_count": result.get("document_count", 0),
            "total_documents": result.get("total_documents_in_db", 0),
            "file_path": result.get("file_path", temp_path)
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/query")
async def query_document(request: QuestionRequest):
    """
    질의 시 documentId가 있으면 해당 문서만 검색, 없으면 전체 문서에서 검색합니다.
    """
    try:
        # document_id가 빈 문자열이면 None으로 변환
        document_id = request.document_id if request.document_id else None
        
        # QueryService를 통해 질의 처리
        response = await query_service.process_query(
            question=request.question,
            document_id=document_id
        )
        
        if response["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=response["error"]
            )

        return {
            "answer": clean_text(response["answer"]),
            "document_id": document_id,
            "query_type": response.get("query_type", "unknown"),
            "source_documents": [
                {
                    "content": clean_text(doc.get("content", str(doc)) if isinstance(doc, dict) else (doc.page_content if hasattr(doc, 'page_content') else str(doc))),
                    "metadata": {
                        key: clean_text(str(value)) if isinstance(value, str) else value
                        for key, value in (
                            doc.get("metadata", {}) if isinstance(doc, dict) else 
                            (doc.metadata if hasattr(doc, 'metadata') else {})
                        ).items()
                    }
                }
                for doc in response.get("source_documents", [])
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 