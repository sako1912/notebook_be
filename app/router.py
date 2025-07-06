from fastapi import APIRouter
from app.api.files import router as files_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router

# 메인 라우터 생성
api_router = APIRouter()

# 각 모듈의 라우터들을 메인 라우터에 포함
api_router.include_router(files_router, prefix="/files", tags=["files"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])

# 라우터 목록 (필요시 사용)
__all__ = ["api_router"]
