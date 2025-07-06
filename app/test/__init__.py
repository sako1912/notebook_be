"""
RAG 서비스 테스트 모듈

이 패키지는 RAG 서비스의 다양한 테스트와 예시를 포함합니다.

테스트 파일들:
- test_rag_service.py: 기본 기능 테스트
- simple_test.py: 간단한 통합 테스트
- working_example.py: 실제 동작 예시
- how_to_use.py: 사용법 가이드
- example_usage.py: 상세한 사용 예시 및 시나리오
- check_available_models.py: AWS Bedrock 모델 확인

실행 방법:
PYTHONPATH=/Users/sako/project/notebook_be python app/test/파일명.py
"""

__version__ = "1.0.0"
__author__ = "RAG Service Team"

# 테스트 실행 도우미
def get_project_root():
    """프로젝트 루트 경로 반환"""
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_path():
    """Python 경로 설정"""
    import sys
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.append(project_root) 