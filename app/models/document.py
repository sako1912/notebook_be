from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class DocumentMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_name: str
    filename: str
    file_size: int
    content_type: str
    s3_key: str
    status: str = "uploaded"  # uploaded, processing, completed, error
    chunk_count: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DocumentRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    chat_history: Optional[List[tuple[str, str]]] = []

class DocumentSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    file_types: Optional[List[str]] = None

class BatchProcessRequest(BaseModel):
    folder_path: str
    file_extensions: Optional[List[str]] = None 