from typing import Dict, Any, List
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import SemanticChunker
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pathlib import Path
import os
import pickle
from app.core.config import get_settings

settings = get_settings()

# 한국어 프롬프트 템플릿 정의
CONDENSE_QUESTION_TEMPLATE = """주어진 대화 기록을 참고하여 최신 질문에 대한 독립적인 질문을 만들어주세요.

대화 기록: {chat_history}
최신 질문: {question}

독립적인 질문:"""

QA_TEMPLATE = """아래 제공된 컨텍스트를 사용하여 질문에 답변해주세요. 
컨텍스트에서 답을 찾을 수 없다면, "주어진 문서에서 해당 정보를 찾을 수 없습니다."라고 답변해주세요.
답변은 한국어로 해주세요.

컨텍스트: {context}

질문: {question}

답변:"""

class RAGService:
    def __init__(self):
        # 캐시 저장소 설정
        cache_dir = os.path.join(settings.base_dir, "embedding_cache")
        fs = LocalFileStore(cache_dir)
        
        # 기본 임베딩 모델
        underlying_embeddings = OpenAIEmbeddings()
        
        # 캐시 지원 임베딩 생성
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            fs,
            namespace="openai_embeddings_cache"
        )
        
        # SemanticChunker 초기화 - 캐시된 임베딩 사용
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold=0.3,  # 의미적 유사도 임계값 (낮을수록 더 작은 청크)
            min_chunk_size=200,        # 최소 청크 크기
            max_chunk_size=1000,       # 최대 청크 크기
            breakpoint_window_size=3   # 문장 간 유사도 비교 윈도우 크기
        )
        
        self.persist_dir = settings.faiss_index_dir
        self.vectorstore = None
        self.qa_chain = None
        
        # 기존 인덱스가 있다면 로드
        index_path = os.path.join(self.persist_dir, "faiss_index.pkl")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.vectorstore = pickle.load(f)
            self._initialize_qa_chain()

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서를 처리하고 벡터 저장소에 저장합니다.
        """
        try:
            # 문서 로드 및 청크 분할
            loader = self._get_loader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # 청크 정보 로깅
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            # 벡터 저장소 생성 또는 업데이트
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # FAISS 인덱스 저장
            index_path = os.path.join(self.persist_dir, "faiss_index.pkl")
            with open(index_path, "wb") as f:
                pickle.dump(self.vectorstore, f)
            
            # QA 체인 초기화
            self._initialize_qa_chain()
            
            return {
                "status": "success",
                "chunk_count": len(chunks),
                "average_chunk_size": int(avg_size),
                "file_path": file_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }

    def query(self, question: str, chat_history: List = None) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.
        1. 질문을 벡터화
        2. 유사한 문서 검색
        3. LLM으로 답변 생성
        """
        if self.qa_chain is None:
            return {
                "status": "error",
                "error": "문서가 로드되지 않았습니다. 먼저 문서를 처리해주세요."
            }
            
        try:
            chat_history = chat_history or []
            
            # 1. 질문 벡터화 및 관련 문서 검색
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.7
                }
            )
            
            # 2. QA 체인 실행 (벡터화 -> 검색 -> 답변 생성)
            response = self.qa_chain({
                "question": question, 
                "chat_history": chat_history
            })
            
            # 3. 검색된 문서와 답변 반환
            return {
                "status": "success",
                "answer": response["answer"],
                "source_documents": response["source_documents"],
                "relevant_chunks": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)  # 유사도 점수
                    }
                    for doc in response["source_documents"]
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _get_loader(self, file_path: str):
        """
        파일 확장자에 따른 적절한 로더를 반환합니다.
        """
        file_extension = Path(file_path).suffix.lower()
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader
        }
        
        loader_class = loaders.get(file_extension)
        if not loader_class:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
            
        if file_extension == '.txt':
            return loader_class(file_path, encoding='utf-8')
        return loader_class(file_path)

    def _initialize_qa_chain(self):
        """
        QA 체인을 초기화합니다.
        """
        # 1. LLM 초기화
        llm = ChatOpenAI(
            temperature=0,  # 결정적인 답변을 위해 temperature를 0으로 설정
            model_name="gpt-4"  # 더 정확한 답변을 위해 GPT-4 사용
        )
        
        # 2. 프롬프트 템플릿 설정
        condense_question_prompt = PromptTemplate(
            template=CONDENSE_QUESTION_TEMPLATE,
            input_variables=["chat_history", "question"]
        )
        
        qa_prompt = PromptTemplate(
            template=QA_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # 3. 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 4. QA 체인 생성
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,  # 상위 4개의 관련 문서 검색
                    "score_threshold": 0.7  # 유사도 점수 임계값
                }
            ),
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True  # 처리 과정 로깅
        )

# 사용 예시:
"""
# RAG 서비스 초기화
rag_service = RAGService()

# 문서 처리
result = rag_service.process_document("example.pdf")
if result["status"] == "success":
    print(f"문서가 성공적으로 처리되었습니다. 청크 수: {result['chunk_count']}")

# 질문하기
chat_history = []
response = rag_service.query("이 문서의 주요 내용은 무엇인가요?", chat_history)
if response["status"] == "success":
    print(f"답변: {response['answer']}")
    # 다음 질문을 위해 대화 기록 업데이트
    chat_history.append(("이 문서의 주요 내용은 무엇인가요?", response["answer"]))
""" 