"""
AWS Bedrock 사용 가능한 모델 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import boto3
from app.core.config import get_settings

def check_available_models():
    """사용 가능한 Bedrock 모델 확인"""
    print("=== AWS Bedrock 사용 가능한 모델 확인 ===\n")
    
    try:
        settings = get_settings()
        
        # Bedrock 클라이언트 생성
        bedrock_client = boto3.client(
            'bedrock',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        
        # 사용 가능한 모델 목록 가져오기
        response = bedrock_client.list_foundation_models()
        
        print("1. 사용 가능한 모든 모델:")
        for model in response['modelSummaries']:
            print(f"   - {model['modelId']}: {model['modelName']}")
        
        print("\n2. Claude 모델:")
        claude_models = [m for m in response['modelSummaries'] if 'anthropic' in m['modelId']]
        for model in claude_models:
            print(f"   - {model['modelId']}: {model['modelName']}")
        
        print("\n3. 임베딩 모델:")
        embedding_models = [m for m in response['modelSummaries'] if 'embed' in m['modelId']]
        for model in embedding_models:
            print(f"   - {model['modelId']}: {model['modelName']}")
            
        # 추천 모델 ID 출력
        print("\n4. 추천 모델 ID:")
        if claude_models:
            print(f"   LLM 모델: {claude_models[0]['modelId']}")
        if embedding_models:
            print(f"   임베딩 모델: {embedding_models[0]['modelId']}")
        
    except Exception as e:
        print(f"❌ 모델 확인 실패: {e}")
        print("\n대안적인 방법:")
        print("1. AWS 콘솔에서 Bedrock 모델 액세스 확인")
        print("2. 다음 기본 모델 ID 시도:")
        print("   - anthropic.claude-3-haiku-20240307-v1:0")
        print("   - anthropic.claude-instant-v1")
        print("   - amazon.titan-text-lite-v1")

if __name__ == "__main__":
    check_available_models() 