from fastapi import APIRouter, UploadFile, File, status, HTTPException
from app.services.s3 import S3Service
from app.services.document_service import DocumentService
from app.models.document import DocumentMetadata
from typing import Dict

router = APIRouter()
s3_service = S3Service()
document_service = DocumentService()

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
    
    # 문서 메타데이터 생성 및 저장
    document = DocumentMetadata(
        original_name=file.filename,
        filename=file.filename,
        file_size=file_size,
        content_type=file.content_type or "application/octet-stream",
        s3_key=file.filename,
        status="uploaded"
    )
    
    document_service.save_document_metadata(document)
    
    return {
        "message": "파일이 성공적으로 업로드되었습니다.",
        "file_url": file_url,
        "document_id": document.id
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