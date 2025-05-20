from fastapi import APIRouter, UploadFile, File, status, HTTPException
from app.services.s3 import S3Service
from typing import Dict

router = APIRouter()
s3_service = S3Service()

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

    # 파일 이름에서 확장자 추출
    file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
    
    # S3에 파일 업로드 및 URL 반환
    file_url = await s3_service.upload_file(file, file.filename)
    
    return {
        "message": "파일이 성공적으로 업로드되었습니다.",
        "file_url": file_url
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