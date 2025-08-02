# RAG 마이크로서비스 아키텍처

## 개요

기존의 단일 `RAGService` 클래스를 마이크로서비스 아키텍처로 재구성하여 각 기능을 독립적인 서비스로 분리했습니다. 이를 통해 확장성, 유지보수성, 및 개별 서비스 최적화가 가능해졌습니다.

## 아키텍처 구조

```
┌─────────────────────────────────────────┐
│           RAGOrchestrator               │
│         (서비스 조합 레이어)              │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌─────────┐ ┌─────────────┐ ┌─────────────┐
│Document │ │ Retrieval   │ │ Generation  │
│Service  │ │ Service     │ │ Service     │
└─────────┘ └─────────────┘ └─────────────┘
    │             │             │
    ▼             ▼             │
┌─────────┐ ┌─────────────┐     │
│Embedding│ │VectorStore  │     │
│Service  │ │Service      │     │
└─────────┘ └─────────────┘     │
    │             │             │
    └─────────────┼─────────────┘
                  ▼
          ┌───────────────┐
          │ 기존 서비스들   │
          │(LLM, Vector,  │
          │PromptManager) │
          └───────────────┘
```

## 서비스별 역할

### 1. DocumentService

**역할**: 문서 처리 전담

- 파일 유효성 검사
- 문서 로드 및 청킹
- 메타데이터 생성

**주요 메서드**:

```python
validate_file(file_path)          # 파일 검증
process_document(file_path)       # 문서 처리
get_supported_extensions()        # 지원 확장자
```

### 2. EmbeddingService

**역할**: 임베딩 생성 및 관리

- 텍스트 벡터화
- 임베딩 캐싱
- Bedrock 연동

**주요 메서드**:

```python
embed_text(text)                  # 단일 텍스트 임베딩
embed_texts(texts)                # 배치 임베딩
get_embedding_dimension()         # 차원 정보
clear_cache()                     # 캐시 초기화
```

### 3. VectorStoreService

**역할**: 벡터 저장소 관리

- 문서 추가/삭제
- 벡터 검색
- 인덱스 최적화

**주요 메서드**:

```python
add_documents(documents)          # 문서 추가
search_documents(query, k)        # 벡터 검색
search_with_scores(query)         # 점수 포함 검색
clear_all_documents()             # 전체 삭제
```

### 4. RetrievalService

**역할**: 고급 검색 기능

- 다양한 검색 전략
- 하이브리드 검색
- 다중 쿼리 검색

**주요 메서드**:

```python
search(query, search_type)        # 통합 검색
multi_query_search(queries)       # 다중 쿼리
get_search_stats()               # 검색 통계
```

**지원 검색 유형**:

- `vector`: 의미 기반 벡터 검색
- `hybrid`: 벡터 + 키워드 검색
- `keyword`: 키워드 기반 검색

### 5. GenerationService

**역할**: LLM 답변 생성

- RAG 기반 답변 생성
- 대화형 답변
- 문서 요약

**주요 메서드**:

```python
generate_answer(question, context)           # RAG 답변
generate_conversational_answer(question)     # 대화형 답변
summarize_documents(documents)               # 문서 요약
```

### 6. RAGOrchestrator

**역할**: 서비스 조합 및 워크플로우 관리

- 전체 RAG 파이프라인 조정
- 서비스 간 데이터 흐름 관리
- 기존 API 호환성 유지

## 사용법

### 기본 사용법 (권장)

```python
from app.services.rag_orchestrator import RAGOrchestrator

# 1. 초기화
rag = RAGOrchestrator()

# 2. 문서 업로드
result = rag.upload_document("document.pdf")

# 3. 검색
search_result = rag.retriever("질문", search_type="hybrid")

# 4. 질문 답변
answer = await rag.ask_question("질문")

# 5. 대화형 답변
chat_answer = await rag.ask_question_conversational(
    "질문",
    chat_history=[("이전질문", "이전답변")]
)

# 6. 문서 요약
summary = await rag.summarize_documents(
    query="특정 주제",
    summary_type="bullet_points"
)
```

### 개별 서비스 사용법 (고급)

```python
# 각 서비스를 독립적으로 사용
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

# 의존성 주입으로 서비스 연결
doc_service = DocumentService()
embedding_service = EmbeddingService()
vector_service = VectorStoreService(embedding_service)

# 개별 서비스 호출
doc_result = doc_service.process_document("file.pdf")
vector_service.add_documents(doc_result["documents"])
```

## 장점

### 1. **독립적 확장성**

- 각 서비스를 개별적으로 스케일링 가능
- 리소스 요구사항에 따른 선택적 확장

### 2. **모듈화**

- 서비스별 독립적 개발 및 테스트
- 개별 서비스 교체 용이성

### 3. **장애 격리**

- 한 서비스의 장애가 전체 시스템에 미치는 영향 최소화
- 서비스별 독립적 장애 복구

### 4. **기술 스택 다양성**

- 서비스별 최적화된 기술 선택 가능
- 점진적 기술 스택 업그레이드

### 5. **개발 생산성**

- 팀별 독립적 개발 가능
- 서비스별 전문화

## 기존 코드와의 호환성

기존 `RAGService`를 사용하던 코드는 `RAGOrchestrator`로 간단히 교체 가능:

```python
# 기존
from app.services.rag_service import RAGService
rag = RAGService()

# 새로운 구조
from app.services.rag_orchestrator import RAGOrchestrator
rag = RAGOrchestrator()

# 동일한 API 사용 가능
result = rag.upload_document("file.pdf")
answer = await rag.ask_question("질문")
```

## 모니터링 및 상태 확인

```python
# 전체 서비스 상태 확인
status = rag.get_service_status()

# 개별 서비스 상태
vector_status = rag.get_database_status()
embedding_info = rag.embedding_service.get_cache_info()
```

## 확장 예시

### 새로운 검색 전략 추가

```python
# RetrievalService에 새로운 검색 메서드 추가
def _semantic_search(self, query, k):
    # 새로운 검색 로직 구현
    pass
```

### 새로운 LLM 모델 연동

```python
# GenerationService에 새로운 생성 메서드 추가
async def generate_with_custom_model(self, question, context):
    # 새로운 모델 사용 로직
    pass
```

## 배포 고려사항

### 단일 프로세스 배포

현재 구조는 단일 프로세스에서 모든 서비스가 실행되지만, 향후 다음과 같이 확장 가능:

### 분산 배포

- 각 서비스를 독립적인 컨테이너로 배포
- API Gateway를 통한 서비스 라우팅
- 서비스 간 통신을 위한 gRPC/REST API 연동

### 클라우드 네이티브

- Kubernetes를 활용한 서비스 오케스트레이션
- 서비스 메시(Istio) 도입
- 분산 추적 및 모니터링

이 새로운 아키텍처는 현재의 요구사항을 충족하면서도 미래의 확장성을 고려한 설계입니다.
