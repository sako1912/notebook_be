from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.rag_service import RAGService
from app.services.s3 import S3Service
from app.services.document_service import DocumentService
from app.models.document import DocumentRequest
import os
import re
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()
rag_service = RAGService()
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
    질의 시 documentId가 있으면 해당 문서만 검색, 없으면 전체 문서에서 검색합니다.
    """
    try:
        # document_id가 빈 문자열이면 None으로 변환
        document_id = request.document_id if request.document_id else None
        
        # documentId가 있으면 해당 문서가 존재하는지 확인
        if document_id:
            document = document_service.get_document_by_id(document_id)
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
        
        response = await rag_service.query(request.question, document_id=document_id)
        print(f"response:: {response}")
        #response:: {'status': 'success', 'answer': '안녕하세요! 반갑습니다. 오늘 하루는 어떠신가요? 무엇을 도와드릴까요?', 'source_documents': [], 'relevant_chunks': [], 'query_type': 'general'}
        
        if response["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=response["error"]
            )

        return {
            "answer": clean_text(response["answer"]),
            "source_documents": [
                {
                    "content": clean_text(doc.page_content),
                    "metadata": {
                        key: clean_text(str(value)) if isinstance(value, str) else value
                        for key, value in doc.metadata.items()
                    }
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