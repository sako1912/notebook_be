"""
RAG 서비스 실제 사용 방법

이 파일은 리팩토링된 RAG 서비스를 실제로 사용하는 방법을 보여줍니다.
"""

import os
import sys
import asyncio

# PYTHONPATH 설정 (프로젝트 루트 디렉토리 추가)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

async def main():
    """실제 사용 예시"""
    
    print("=== RAG 서비스 실제 사용법 ===\n")
    
    # 1. 환경 설정 확인
    print("1. 환경 설정 확인:")
    print("   - AWS_ACCESS_KEY_ID:", "✅ 설정됨" if os.getenv('AWS_ACCESS_KEY_ID') else "❌ 설정 필요")
    print("   - AWS_SECRET_ACCESS_KEY:", "✅ 설정됨" if os.getenv('AWS_SECRET_ACCESS_KEY') else "❌ 설정 필요")
    print("   - AWS_REGION:", os.getenv('AWS_REGION', '❌ 설정 필요'))
    
    # 2. 서비스 초기화
    print("\n2. 서비스 초기화:")
    try:
        from app.services import RAGService
        
        # 주의: AWS 설정이 필요합니다
        rag_service = RAGService()
        print("   ✅ RAG 서비스 초기화 성공")
        
        # 3. 지원하는 파일 형식 확인
        print(f"\n3. 지원하는 파일 형식: {rag_service.get_supported_extensions()}")
        
        # 4. 문서 처리 예시
        print("\n4. 문서 처리 방법:")
        print("   # 방법 1: 특정 문서 ID로 처리")
        print("   result = rag_service.process_document('문서파일.pdf', document_id='내문서_001')")
        print("   ")
        print("   # 방법 2: 전체 인덱스에 추가")
        print("   result = rag_service.process_document('문서파일.pdf')")
        
        # 5. 질문하기 예시
        print("\n5. 질문하기 방법:")
        print("   # 문서 기반 질문")
        print("   response = await rag_service.query('질문내용', document_id='내문서_001')")
        print("   ")
        print("   # 일반 질문 (문서 없이)")
        print("   response = await rag_service.query('일반적인 질문')")
        
        # 6. 대화 기록 포함
        print("\n6. 대화 기록 포함:")
        print("   chat_history = [('이전 질문', '이전 답변')]")
        print("   response = await rag_service.query('질문', chat_history=chat_history)")
        
    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        print("\n   해결 방법:")
        print("   1. AWS 설정 확인")
        print("   2. 인터넷 연결 확인")
        print("   3. AWS Bedrock 권한 확인")

def show_usage_commands():
    """사용 명령어 표시"""
    print("\n=== 실행 명령어 ===")
    print("1. 기본 테스트:")
    print("   PYTHONPATH=/Users/sako/project/notebook_be python app/test/test_rag_service.py")
    print()
    print("2. 이 사용법 파일 실행:")
    print("   PYTHONPATH=/Users/sako/project/notebook_be python app/test/how_to_use.py")
    print()
    print("3. 실제 사용 시 (Python 스크립트에서):")
    print("   import sys")
    print("   import os")
    print("   # 프로젝트 루트 경로 추가")
    print("   sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))")
    print("   from app.services import RAGService")
    print()
    print("4. 또는 환경 변수 설정:")
    print("   export PYTHONPATH=/Users/sako/project/notebook_be")
    print("   python your_script.py")

if __name__ == "__main__":
    asyncio.run(main())
    show_usage_commands() 