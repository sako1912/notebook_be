from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.files import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(title="Notebook ML API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 도메인을 지정해야 합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"]) 