from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
import logging

# PPT 파일 처리를 위한 임포트
try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    다양한 문서 형식을 로드하는 클래스
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.ppt', '.pptx'}
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """파일 확장자가 지원되는지 확인"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        파일 경로에서 문서를 로드하여 Document 객체 리스트로 반환
        """
        if not cls.is_supported(file_path):
            raise ValueError(f"지원하지 않는 파일 형식입니다: {Path(file_path).suffix}")
        
        try:
            loader = cls._get_loader(file_path)
            documents = loader.load()
            
            logger.info(f"문서 로드 완료: {file_path}, 페이지 수: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"문서 로드 실패: {file_path}, 오류: {str(e)}")
            raise
    
    @classmethod
    def _get_loader(cls, file_path: str):
        """파일 확장자에 따른 적절한 로더 반환"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.docx':
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        elif file_extension in ['.ppt', '.pptx']:
            return cls._get_ppt_loader(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
    
    @classmethod
    def _get_ppt_loader(cls, file_path: str):
        """PPT 파일을 위한 커스텀 로더"""
        if not PPTX_AVAILABLE:
            raise ValueError("PPT 파일 처리를 위해 python-pptx 라이브러리가 필요합니다.")
        
        class PPTLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            
            def load(self):
                prs = Presentation(self.file_path)
                documents = []
                
                for slide_num, slide in enumerate(prs.slides):
                    slide_text = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_text.append(shape.text.strip())
                    
                    if slide_text:
                        content = "\n".join(slide_text)
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "source": self.file_path,
                                "slide_number": slide_num + 1
                            }
                        ))
                
                return documents
        
        return PPTLoader(file_path) 