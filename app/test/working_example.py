"""
실제 동작하는 RAG 서비스 예시
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services import RAGService
import asyncio

async def working_example():
    """실제 동작하는 예시"""
    print("=== RAG 서비스 실제 동작 예시 ===\n")
    
    # 1. 서비스 초기화
    print("1. 서비스 초기화...")
    rag_service = RAGService()
    print("   ✅ 초기화 완료\n")
    
    # 2. 테스트 문서 처리
    print("2. 테스트 문서 처리...")
    test_file = "test_document.txt"
    
    if os.path.exists(test_file):
        result = rag_service.process_document(test_file, document_id="test_doc")
        print(f"   결과: {result}")
        
        if result["status"] == "success":
            print("   ✅ 문서 처리 성공!\n")
            
            # 3. 문서 기반 질문
            print("3. 문서 기반 질문...")
            question = "이 문서의 주요 내용은 무엇인가요?"
            
            response = await rag_service.query(question, document_id="test_doc")
            
            if response["status"] == "success":
                print(f"   질문: {question}")
                print(f"   답변: {response['answer']}")
                print(f"   타입: {response['query_type']}")
                print(f"   관련 문서 수: {len(response['relevant_chunks'])}")
                print("   ✅ 질문 성공!\n")
            else:
                print(f"   ❌ 질문 실패: {response['error']}\n")
            
            # 4. 일반 질문 (문서 없이)
            print("4. 일반 질문 (문서 없이)...")
            general_question = "안녕하세요! 오늘 날씨는 어때요?"
            
            response = await rag_service.query(general_question)
            
            if response["status"] == "success":
                print(f"   질문: {general_question}")
                print(f"   답변: {response['answer']}")
                print(f"   타입: {response['query_type']}")
                print("   ✅ 일반 질문 성공!\n")
            else:
                print(f"   ❌ 일반 질문 실패: {response['error']}\n")
                
        else:
            print(f"   ❌ 문서 처리 실패: {result['error']}")
    else:
        print(f"   ❌ 테스트 파일 없음: {test_file}")
        print("   test_document.txt 파일을 프로젝트 루트에 생성해주세요.")

if __name__ == "__main__":
    asyncio.run(working_example()) 