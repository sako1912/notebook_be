"""
실제 사용 가능한 간단한 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services import RAGService
import asyncio

async def simple_test():
    print("=== RAG 서비스 간단 테스트 ===")
    
    try:
        # 1. 서비스 초기화
        print("1. RAG 서비스 초기화 중...")
        rag_service = RAGService()
        print("   ✅ 초기화 완료")
        
        # 2. 지원하는 파일 형식 확인
        print(f"2. 지원하는 파일 형식: {rag_service.get_supported_extensions()}")
        
        # 3. 문서 처리 테스트
        print("3. 문서 처리 테스트...")
        test_file = "test_document.txt"
        
        if os.path.exists(test_file):
            result = rag_service.process_document(test_file, document_id="test_doc")
            print(f"   문서 처리 결과: {result}")
            
            if result["status"] == "success":
                print("   ✅ 문서 처리 성공!")
                
                # 4. 문서 목록 확인
                docs = rag_service.get_document_list()
                print(f"4. 저장된 문서 목록: {docs}")
                
                # 5. 문서 기반 질문 (간단한 질문)
                print("5. 문서 기반 질문 테스트...")
                try:
                    response = await rag_service.query(
                        question="이 문서의 주요 내용은 무엇인가요?",
                        document_id="test_document"
                    )
                    if response["status"] == "success":
                        print(f"   ✅ 답변: {response['answer']}")
                    else:
                        print(f"   ❌ 질문 실패: {response['error']}")
                except Exception as e:
                    print(f"   ❌ 질문 중 오류: {e}")
                
            else:
                print(f"   ❌ 문서 처리 실패: {result['error']}")
        else:
            print(f"   ❌ 테스트 파일 없음: {test_file}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    asyncio.run(simple_test()) 