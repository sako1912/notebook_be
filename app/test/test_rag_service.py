"""
RAG 서비스 기본 기능 테스트

이 파일은 리팩토링된 RAG 서비스의 기본 기능들이 정상적으로 동작하는지 확인하는 테스트입니다.
"""

import asyncio
import tempfile
import os
from pathlib import Path

from app.services import RAGService, DocumentLoader, VectorStoreManager, PromptManager

def test_document_loader():
    """문서 로더 테스트"""
    print("=== 문서 로더 테스트 ===")
    
    # 지원하는 파일 확장자 테스트
    print(f"지원하는 확장자: {DocumentLoader.SUPPORTED_EXTENSIONS}")
    
    # 파일 지원 여부 테스트
    test_files = ["test.pdf", "test.docx", "test.txt", "test.pptx", "test.unknown"]
    for file in test_files:
        supported = DocumentLoader.is_supported(file)
        print(f"{file}: {'지원' if supported else '미지원'}")
    
    # 임시 텍스트 파일 생성 및 로드 테스트
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write("이것은 테스트 문서입니다.\n여러 줄의 내용을 포함합니다.")
        tmp_file_path = tmp_file.name
    
    try:
        documents = DocumentLoader.load_document(tmp_file_path)
        print(f"텍스트 파일 로드 성공: {len(documents)}개 문서")
        print(f"첫 번째 문서 내용: {documents[0].page_content[:50]}...")
    except Exception as e:
        print(f"텍스트 파일 로드 실패: {e}")
    finally:
        os.unlink(tmp_file_path)
    
    print("문서 로더 테스트 완료\n")

def test_prompt_manager():
    """프롬프트 매니저 테스트"""
    print("=== 프롬프트 매니저 테스트 ===")
    
    # 기본 프롬프트 생성 테스트
    question = "안녕하세요!"
    prompt = PromptManager.create_general_prompt(question)
    print(f"기본 프롬프트:\n{prompt}\n")
    
    # 대화 기록 포함 프롬프트 생성 테스트
    chat_history = [
        ("이전 질문", "이전 답변"),
        ("또 다른 질문", "또 다른 답변")
    ]
    prompt_with_history = PromptManager.create_general_prompt(question, chat_history)
    print(f"대화 기록 포함 프롬프트:\n{prompt_with_history}\n")
    
    # 프롬프트 템플릿 테스트
    qa_prompt = PromptManager.get_qa_prompt()
    print(f"QA 프롬프트 템플릿: {qa_prompt.template[:100]}...")
    
    condense_prompt = PromptManager.get_condense_question_prompt()
    print(f"질문 압축 프롬프트 템플릿: {condense_prompt.template[:100]}...")
    
    print("프롬프트 매니저 테스트 완료\n")

def test_vector_store_manager():
    """벡터 저장소 매니저 테스트"""
    print("=== 벡터 저장소 매니저 테스트 ===")
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"임시 디렉토리: {tmp_dir}")
        
        # 임시 임베딩 클래스 (실제 임베딩 없이 테스트)
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def embed_query(self, text):
                return [0.1, 0.2, 0.3]
        
        mock_embeddings = MockEmbeddings()
        
        # VectorStoreManager 초기화 테스트
        try:
            vm = VectorStoreManager(mock_embeddings, tmp_dir)
            print("벡터 저장소 매니저 초기화 성공")
            
            # 문서 ID 목록 조회 테스트
            doc_ids = vm.list_document_ids()
            print(f"초기 문서 ID 목록: {doc_ids}")
            
        except Exception as e:
            print(f"벡터 저장소 매니저 테스트 실패: {e}")
    
    print("벡터 저장소 매니저 테스트 완료\n")

def test_rag_service_initialization():
    """RAG 서비스 초기화 테스트"""
    print("=== RAG 서비스 초기화 테스트 ===")
    
    try:
        # RAG 서비스 초기화 (실제 AWS 연결 없이는 실패할 수 있음)
        rag_service = RAGService()
        print("RAG 서비스 초기화 성공")
        
        # 기본 메서드 테스트
        extensions = rag_service.get_supported_extensions()
        print(f"지원하는 확장자: {extensions}")
        
        doc_list = rag_service.get_document_list()
        print(f"문서 목록: {doc_list}")
        
    except Exception as e:
        print(f"RAG 서비스 초기화 실패: {e}")
        print("이는 AWS 설정이 없거나 네트워크 문제일 수 있습니다.")
    
    print("RAG 서비스 초기화 테스트 완료\n")

async def test_basic_functionality():
    """기본 기능 통합 테스트"""
    print("=== 기본 기능 통합 테스트 ===")
    
    # 모든 개별 테스트 실행
    test_document_loader()
    test_prompt_manager()
    test_vector_store_manager()
    test_rag_service_initialization()
    
    print("모든 기본 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality()) 