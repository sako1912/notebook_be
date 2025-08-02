from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from app.models.document import DocumentMetadata, DocumentSearchRequest, BatchProcessRequest
from app.services.document_service import DocumentService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
document_service = DocumentService()

@router.get("/", response_model=List[DocumentMetadata])
async def get_documents(
    status_filter: Optional[str] = Query(None, description="상태 필터 (uploaded, processing, completed, error)"),
    file_type: Optional[str] = Query(None, description="파일 타입 필터 (pdf, txt, docx, ppt, pptx)")
):
    """문서 목록 조회"""
    try:
        documents = document_service.get_all_documents()
        print(f"documents:: {documents}")
        # 상태 필터 적용
        if status_filter:
            documents = [doc for doc in documents if doc.status == status_filter]
        
        # 파일 타입 필터 적용
        if file_type:
            documents = [doc for doc in documents if doc.filename.lower().endswith(f'.{file_type.lower()}')]
        
        return documents
        
    except Exception as e:
        logger.error(f"문서 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 목록 조회 중 오류 발생: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """특정 문서 정보 조회"""
    try:
        document = document_service.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다."
            )
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 조회 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 조회 중 오류 발생: {str(e)}"
        )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """문서 삭제"""
    try:
        success = document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="문서를 찾을 수 없습니다."
            )
        
        return {"message": "문서가 성공적으로 삭제되었습니다.", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 삭제 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 삭제 중 오류 발생: {str(e)}"
        )

@router.post("/search", response_model=List[DocumentMetadata])
async def search_documents(request: DocumentSearchRequest):
    """문서 검색"""
    try:
        documents = document_service.search_documents(
            query=request.query,
            file_types=request.file_types
        )
        
        return documents
        
    except Exception as e:
        logger.error(f"문서 검색 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 검색 중 오류 발생: {str(e)}"
        )

@router.post("/process/batch")
async def process_batch_documents(request: BatchProcessRequest):
    """S3 폴더의 문서들을 일괄 처리"""
    try:
        logger.info(f"배치 처리 시작: {request.folder_path}")
        
        results = await document_service.process_s3_folder(
            folder_path=request.folder_path,
            file_extensions=request.file_extensions
        )
        
        # 결과 요약
        success_count = len([r for r in results if r["status"] == "success"])
        error_count = len([r for r in results if r["status"] == "error"])
        
        return {
            "message": f"배치 처리 완료: 성공 {success_count}개, 실패 {error_count}개",
            "total_files": len(results),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"배치 처리 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"배치 처리 중 오류 발생: {str(e)}"
        ) 