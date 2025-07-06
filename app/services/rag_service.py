from typing import Dict, Any, List, Optional, Tuple
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.messages import HumanMessage
from pathlib import Path
import os
import boto3
import logging

from app.core.config import get_settings
from app.services.document_loader import DocumentLoader
from app.services.vector_store_manager import VectorStoreManager
from app.services.prompt_manager import PromptManager
from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)
settings = get_settings()

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 서비스
    
    문서 기반 질답과 일반 대화를 모두 지원하는 AI 서비스
    """
    
    def __init__(self):
        """RAG 서비스 초기화"""
        self.bedrock_client = self._initialize_bedrock_client()
        self.embeddings = self._initialize_embeddings()
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embeddings,
            persist_dir=settings.faiss_index_dir
        )
        self.llm = self._initialize_llm()
        
        logger.info("RAG 서비스 초기화 완료")
    
    def _initialize_bedrock_client(self):
        """AWS Bedrock 클라이언트 초기화"""
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화 (캐시 포함)"""
        # 캐시 저장소 설정
        cache_dir = os.path.join(settings.base_dir, "embedding_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fs = LocalFileStore(cache_dir)
        
        # AWS Bedrock 임베딩 모델
        underlying_embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",  # 한국 리전에서 사용 가능
            region_name=settings.aws_region
        )
        
        # 캐시 지원 임베딩 생성
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            fs,
            namespace="bedrock_embeddings_cache"
        )
    
    def _initialize_llm(self):
        """LLM 모델 초기화"""
        return ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Claude 3.5 Sonnet 사용
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 4000
            }
        )
    
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        문서를 처리하여 벡터 저장소에 저장
        
        Args:
            file_path: 처리할 문서 파일 경로
            document_id: 문서 ID (선택사항)
        
        Returns:
            처리 결과 정보
        """
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": f"파일이 존재하지 않습니다: {file_path}"
                }
            
            # 지원하는 파일 형식인지 확인
            if not DocumentLoader.is_supported(file_path):
                return {
                    "status": "error",
                    "error": f"지원하지 않는 파일 형식입니다: {Path(file_path).suffix}"
                }
            
            # 문서 로드
            documents = DocumentLoader.load_document(file_path)
            
            if not documents:
                return {
                    "status": "error",
                    "error": "문서에서 텍스트를 추출할 수 없습니다"
                }
            
            # 벡터 저장소 생성
            result = self.vector_store_manager.create_vectorstore(
                documents=documents,
                document_id=document_id
            )
            
            # 결과에 파일 경로 추가
            if result["status"] == "success":
                result["file_path"] = file_path
                result["document_count"] = len(documents)
                logger.info(f"문서 처리 완료: {file_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"문서 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }
    
    async def query(
        self, 
        question: str, 
        chat_history: List[Tuple[str, str]] = None, 
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            chat_history: 대화 기록 [(질문, 답변), ...]
            document_id: 특정 문서 ID (선택사항)
        
        Returns:
            답변 결과
        """
        try:
            chat_history = chat_history or []
            
            # 벡터 저장소 로드
            vectorstore = self.vector_store_manager.load_vectorstore(document_id)
            
            if vectorstore is not None:
                # RAG 방식: 문서 기반 질답
                return await self._process_rag_query(question, chat_history, vectorstore)
            else:
                # 일반 대화 방식
                return await self._process_general_query(question, chat_history)
                
        except Exception as e:
            error_msg = f"질의 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    async def _process_rag_query(self, question: str, chat_history: List[Tuple[str, str]], vectorstore) -> Dict[str, Any]:
        """RAG 방식으로 질문 처리"""
        try:
            # QA 체인 생성
            qa_chain = self._create_qa_chain(vectorstore)
            
            # 질문 실행
            response = await qa_chain.ainvoke({
                "question": question,
                "chat_history": chat_history
            })
            
            logger.info(f"RAG 질의 처리 완료: {question[:50]}...")
            
            return {
                "status": "success",
                "answer": response["answer"],
                "source_documents": response.get("source_documents", []),
                "relevant_chunks": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)
                    }
                    for doc in response.get("source_documents", [])
                ],
                "query_type": "rag"
            }
            
        except Exception as e:
            raise Exception(f"RAG 질의 처리 실패: {str(e)}")
    
    async def _process_general_query(self, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        """일반 대화 방식으로 질문 처리"""
        try:
            # 프롬프트 생성
            prompt = PromptManager.create_general_prompt(question, chat_history)
            
            # LLM 실행
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            logger.info(f"일반 질의 처리 완료: {question[:50]}...")
            
            return {
                "status": "success",
                "answer": response.content,
                "source_documents": [],
                "relevant_chunks": [],
                "query_type": "general"
            }
            
        except Exception as e:
            raise Exception(f"일반 질의 처리 실패: {str(e)}")
    
    def _create_qa_chain(self, vectorstore):
        """RAG를 위한 QA 체인 생성 (메모리 없는 방식)"""
        
        # ConversationalRetrievalChain을 메모리 없이 생성
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,  # 상위 4개 문서 검색
                    "score_threshold": 0.3  # 유사도 점수 임계값 (더 관대하게)
                }
            ),
            condense_question_prompt=PromptManager.get_condense_question_prompt(),
            combine_docs_chain_kwargs={"prompt": PromptManager.get_qa_prompt()},
            return_source_documents=True,
            verbose=False
        )
    
    def get_document_list(self) -> List[str]:
        """저장된 문서 ID 목록 반환"""
        return self.vector_store_manager.list_document_ids()
    
    def delete_document(self, document_id: str) -> bool:
        """특정 문서 삭제"""
        return self.vector_store_manager.delete_vectorstore(document_id)
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return list(DocumentLoader.SUPPORTED_EXTENSIONS)

# 사용 예시
"""
# RAG 서비스 초기화
rag_service = RAGService()

# 문서 처리 (특정 문서 ID로)
result = rag_service.process_document("example.pdf", document_id="doc_001")
print(f"처리 결과: {result}")

# 문서 기반 질문
response = await rag_service.query(
    question="이 문서의 주요 내용은 무엇인가요?",
    document_id="doc_001"
)
print(f"답변: {response['answer']}")

# 일반 질문 (문서 없이)
response = await rag_service.query(
    question="안녕하세요. 오늘 날씨가 어때요?"
)
print(f"답변: {response['answer']}")

# 문서 목록 조회
documents = rag_service.get_document_list()
print(f"저장된 문서들: {documents}")
""" 