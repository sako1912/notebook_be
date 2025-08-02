from fastapi import APIRouter, UploadFile, File, status, HTTPException
from app.services.s3 import S3Service
from app.services.document_service import DocumentService
from app.services.rag_orchestrator import RAGOrchestrator
from typing import Dict, Optional, Union
from pydantic import BaseModel
import uuid
from datetime import datetime
import os
import tempfile


class RAGUploadResponse(BaseModel):
    status: str
    message: str
    file_url: str
    document_id: Optional[str] = None
    chunks_processed: Optional[int] = None
    total_documents: Optional[int] = None

class ErrorResponse(BaseModel):
    status: str
    message: str
    file_url: Optional[str] = None

router = APIRouter()
s3_service = S3Service()
document_service = DocumentService()
rag_orchestrator = RAGOrchestrator()

@router.post("/upload/", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def upload_file(file: UploadFile = File(...)):
    """
    파일을 업로드하고 S3에 저장하는 엔드포인트
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일이 제공되지 않았습니다."
        )

    # 파일 크기 확인
    file_content = await file.read()
    file_size = len(file_content)
    
    # 파일 포인터 리셋
    await file.seek(0)
    
    # 파일 이름에서 확장자 추출
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    
    # 지원하는 파일 형식 확인
    supported_extensions = ['pdf', 'txt', 'docx', 'ppt', 'pptx']
    if file_extension.lower() not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
        )
    
    # S3에 파일 업로드 및 URL 반환
    file_url = await s3_service.upload_file(file, file.filename)
    
    # 간단한 메타데이터 생성
    document_id = str(uuid.uuid4())
    
    return {
        "message": "파일이 성공적으로 업로드되었습니다.",
        "file_url": file_url,
        "document_id": document_id
    }

@router.delete("/delete/{filename}", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def delete_file(filename: str):
    """
    S3에서 파일을 삭제하는 엔드포인트
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일 이름이 제공되지 않았습니다."
        )

    await s3_service.delete_file(filename)
    
    return {
        "message": "파일이 성공적으로 삭제되었습니다.",
        "filename": filename
    } 

@router.post("/upload-to-rag/", response_model=Union[RAGUploadResponse, ErrorResponse], status_code=status.HTTP_200_OK)
async def upload_file_to_rag(file: UploadFile = File(...)):
    """
    파일을 업로드하고 S3에 저장한 후, RAG 시스템에 임베딩하여 벡터 DB에 저장하는 엔드포인트
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="파일이 제공되지 않았습니다."
        )

    # 파일 크기 확인
    file_content = await file.read()
    file_size = len(file_content)
    
    # 파일 포인터 리셋
    await file.seek(0)
    
    # 파일 이름에서 확장자 추출
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    
    # 지원하는 파일 형식 확인
    supported_extensions = ['pdf', 'txt', 'docx', 'ppt', 'pptx']
    if file_extension.lower() not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_extensions)}"
        )
    
    try:
        # S3에 파일 업로드
        file_url = await s3_service.upload_file(file, file.filename)
        
        # 통합된 메타데이터 생성
        metadata = {
            "document_id": str(uuid.uuid4()),
            "original_name": file.filename,
            "filename": file.filename,
            "file_size": file_size,
            "content_type": file.content_type or "application/octet-stream",
            "s3_key": file.filename,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # S3에서 파일 다운로드
        file_content = await s3_service.download_file(file.filename)
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        try:
            # RAG 시스템에 문서 추가
            rag_result = await rag_orchestrator.upload_document(
                file_path=temp_path,
                metadata=metadata
            )
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if rag_result["status"] == "error":
            metadata["status"] = "error"
            metadata["error_message"] = rag_result["error"]
            return ErrorResponse(
                status="error",
                message=f"RAG 처리 중 오류 발생: {rag_result['error']}",
                file_url=file_url
            )
        
        metadata["status"] = "completed"
        metadata["chunk_count"] = rag_result["document_count"]
        
        return RAGUploadResponse(
            status="success",
            message="파일이 성공적으로 업로드되고 RAG 시스템에 추가되었습니다.",
            file_url=file_url,
            document_id=metadata["document_id"],
            chunks_processed=rag_result["document_count"],
            total_documents=rag_result["total_documents_in_db"]
        )

    except Exception as e:
        return ErrorResponse(
            status="error",
            message=f"파일 처리 중 오류 발생: {str(e)}",
            file_url=file_url if 'file_url' in locals() else None
        ) 

