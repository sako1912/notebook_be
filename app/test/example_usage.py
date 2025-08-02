"""
RAG 서비스 사용 예시

이 파일은 리팩토링된 RAG 서비스의 사용법을 보여주는 예시입니다.
실제 사용할 때는 이 코드를 참고하여 구현하세요.
"""

import sys
import os
import asyncio

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services import RAGService

async def main():
    # RAG 서비스 초기화
    print("RAG 서비스 초기화 중...")
    rag_service = RAGService()
    
    # 지원하는 파일 형식 확인
    print(f"지원하는 파일 형식: {rag_service.get_supported_extensions()}")
    
    # 문서 처리 예시
    print("\n=== 문서 처리 예시 ===")
    
    # 방법 1: 특정 문서 ID로 처리
    result = rag_service.process_document(
        file_path="example.pdf", 
        document_id="my_document_001"
    )
    print(f"문서 처리 결과: {result}")
    
    # 방법 2: 전체 인덱스에 추가 (document_id 없이)
    result = rag_service.process_document("another_document.pdf")
    print(f"전체 인덱스 처리 결과: {result}")
    
    # 저장된 문서 목록 확인
    print(f"\n저장된 문서 목록: {rag_service.get_document_list()}")
    
    # 질문하기 예시
    print("\n=== 질문하기 예시 ===")
    
    # 1. 특정 문서에 대한 질문
    print("1. 특정 문서에 대한 질문")
    response = await rag_service.query(
        question="이 문서의 주요 내용은 무엇인가요?",
        document_id="my_document_001"
    )
    print(f"문서 기반 답변: {response['answer']}")
    print(f"답변 타입: {response['query_type']}")
    
    # 2. 전체 문서에 대한 질문
    response = await rag_service.query(
        question="모든 문서에서 가장 중요한 키워드는 무엇인가요?"
    )
    print(f"전체 문서 기반 답변: {response['answer']}")
    
    # 3. 대화 기록을 포함한 질문
    chat_history = [
        ("이전에 무엇에 대해 물어봤나요?", "문서의 주요 내용에 대해 물어보셨습니다."),
        ("그럼 추가로 알고 싶은 게 있어요", "네, 무엇을 더 알고 싶으신가요?")
    ]
    
    response = await rag_service.query(
        question="세부 내용을 좀 더 설명해주세요",
        chat_history=chat_history,
        document_id="my_document_001"
    )
    print(f"대화 맥락 포함 답변: {response['answer']}")
    
    # 4. 문서 없이 일반 질문
    print("4. 문서 없이 일반 질문")
    response = await rag_service.query(
        question="안녕! 너는 어떤 ai인지 설명해줘(누가 만들었고 어떤 모델명인지 등)"
    )
    print(f"일반 질문 답변: {response['answer']}")
    print(f"답변 타입: {response['query_type']}")
    
    # 답변 상세 정보 확인
    print("\n=== 답변 상세 정보 ===")
    if response['relevant_chunks']:
        print("관련 문서 청크:")
        for i, chunk in enumerate(response['relevant_chunks'][:2]):  # 상위 2개만 출력
            print(f"  청크 {i+1}:")
            print(f"    내용: {chunk['content'][:100]}...")
            print(f"    메타데이터: {chunk['metadata']}")
    
    # 문서 삭제 예시
    print("\n=== 문서 삭제 예시 ===")
    success = rag_service.delete_document("my_document_001")
    print(f"문서 삭제 성공: {success}")
    
    print(f"삭제 후 문서 목록: {rag_service.get_document_list()}")

if __name__ == "__main__":
    asyncio.run(main()) 