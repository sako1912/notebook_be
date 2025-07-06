from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.rag_service import RAGService
from app.services.s3 import S3Service
from app.services.document_service import DocumentService
from app.models.document import DocumentRequest
import os
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()
rag_service = RAGService()
s3_service = S3Service()
document_service = DocumentService()

class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    chat_history: Optional[List[tuple[str, str]]] = []

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

        # RAG 처리
        result = rag_service.process_document(temp_path)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"문서 처리 중 오류 발생: {result['error']}"
            )

        return {
            "message": "문서가 성공적으로 처리되었습니다.",
            "chunk_count": result["chunk_count"],
            "file_path": result["file_path"]
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
    처리된 문서에 대해 질문합니다.
    documentId가 있으면 해당 문서만 검색, 없으면 전체 문서 검색
    """
    try:
        # documentId가 있으면 해당 문서가 존재하는지 확인
        if request.document_id:
            document = document_service.get_document_by_id(request.document_id)
            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="문서를 찾을 수 없습니다."
                )
            
            if document.status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"문서 처리가 완료되지 않았습니다. 현재 상태: {document.status}"
                )
        
        response = await rag_service.query(request.question, request.chat_history, request.document_id)
        
        if response["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=response["error"]
            )

        return {
            "answer": response["answer"],
            "document_id": request.document_id,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["source_documents"]
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 