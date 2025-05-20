from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict
from pathlib import Path
import os
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    s3_bucket_name: str
    
    # 파일 저장 경로 설정
    base_dir: str = str(Path(__file__).resolve().parent.parent.parent)  # project root
    upload_dir: str = os.path.join(base_dir, "uploads")
    faiss_index_dir: str = os.path.join(base_dir, "faiss_index")

    model_config = ConfigDict(
        env_file=".env",
        extra='ignore'  # 추가 필드 무시
    )

    def ensure_directories(self):
        """필요한 디렉토리가 없는 경우에만 생성합니다."""
        directories = [self.upload_dir, self.faiss_index_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"디렉토리 생성됨: {directory}")
            else:
                logger.info(f"기존 디렉토리 사용: {directory}")

@lru_cache()
def get_settings():
    settings = Settings()
    settings.ensure_directories()
    return settings