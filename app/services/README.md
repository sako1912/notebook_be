# RAG 서비스 리팩토링

## 개요

기존의 복잡하고 모놀리식한 RAG 서비스 코드를 더 깔끔하고 유지보수하기 쉬운 모듈화된 구조로 리팩토링했습니다.

## 리팩토링 후 구조

### 1. 모듈 분리

기존의 하나의 큰 클래스를 4개의 독립적인 모듈로 분리했습니다:

#### `document_loader.py`

- **책임**: 다양한 문서 형식(PDF, DOCX, TXT, PPT) 로드
- **주요 기능**:
  - 파일 형식 지원 여부 확인
  - 문서 로드 및 Document 객체 변환
  - PPT 파일 처리를 위한 커스텀 로더

#### `prompt_manager.py`

- **책임**: 프롬프트 템플릿 관리
- **주요 기능**:
  - 한국어 프롬프트 템플릿 제공
  - 대화 기록 포함 프롬프트 생성
  - RAG 및 일반 질답용 프롬프트 분리

#### `vector_store_manager.py`

- **책임**: 벡터 저장소 관리
- **주요 기능**:
  - FAISS 벡터 저장소 생성 및 저장
  - 문서별 독립적인 인덱스 관리
  - 벡터 저장소 로드 및 삭제

#### `rag_service.py`

- **책임**: 전체 RAG 서비스 오케스트레이션
- **주요 기능**:
  - AWS Bedrock 연동
  - 문서 기반 질답 및 일반 대화
  - 서비스 전체 조정 및 관리

### 2. 개선된 특징

#### 🎯 **단일 책임 원칙 (SRP)**

- 각 클래스가 하나의 명확한 책임만 가짐
- 코드 변경 시 영향 범위 최소화

#### 🔧 **높은 응집도, 낮은 결합도**

- 각 모듈이 독립적으로 테스트 가능
- 인터페이스를 통한 느슨한 결합

#### 📝 **향상된 에러 처리**

- 일관된 에러 처리 방식
- 상세한 로깅 시스템

#### 🧪 **테스트 가능성**

- 각 모듈별 독립적인 테스트 가능
- Mock 객체 사용 가능

## 사용법

### 기본 사용법

```python
from app.services import RAGService

# 서비스 초기화
rag_service = RAGService()

# 문서 처리
result = rag_service.process_document(
    file_path="document.pdf",
    document_id="doc_001"
)

# 질문하기
response = await rag_service.query(
    question="이 문서의 주요 내용은 무엇인가요?",
    document_id="doc_001"
)
```

### 개별 모듈 사용법

```python
from app.services import DocumentLoader, PromptManager

# 문서 로더 사용
documents = DocumentLoader.load_document("file.pdf")

# 프롬프트 생성
prompt = PromptManager.create_general_prompt(
    question="안녕하세요!",
    chat_history=[("이전 질문", "이전 답변")]
)
```

## 주요 개선 사항

### 1. 코드 가독성 향상

- 명확한 클래스명과 메서드명
- 상세한 문서화 주석
- 타입 힌트 강화

### 2. 유지보수성 개선

- 모듈화된 구조
- 중복 코드 제거
- 일관된 코딩 스타일

### 3. 확장성 강화

- 새로운 문서 형식 추가 용이
- 새로운 임베딩 모델 연동 가능
- 다양한 벡터 저장소 지원 가능

### 4. 성능 최적화

- 캐시 기반 임베딩
- 효율적인 벡터 저장소 관리
- 메모리 사용량 최적화

## 파일 구조

```
app/services/
├── __init__.py              # 모듈 초기화
├── rag_service.py           # 메인 RAG 서비스
├── document_loader.py       # 문서 로더
├── prompt_manager.py        # 프롬프트 관리
├── vector_store_manager.py  # 벡터 저장소 관리
└── README.md               # 이 파일

app/test/                   # 테스트 및 예시 파일들
├── __init__.py              # 테스트 모듈 초기화
├── test_rag_service.py      # 기본 기능 테스트
├── simple_test.py           # 간단한 통합 테스트
├── working_example.py       # 실제 동작 예시
├── how_to_use.py            # 사용법 가이드
└── check_available_models.py # AWS Bedrock 모델 확인
```

## 테스트 실행

```bash
# 기본 기능 테스트
PYTHONPATH=/Users/sako/project/notebook_be python app/test/test_rag_service.py

# 실제 동작 예시
PYTHONPATH=/Users/sako/project/notebook_be python app/test/working_example.py

# 사용법 가이드
PYTHONPATH=/Users/sako/project/notebook_be python app/test/how_to_use.py

# AWS 모델 확인
PYTHONPATH=/Users/sako/project/notebook_be python app/test/check_available_models.py
```

## 지원하는 파일 형식

- **PDF**: `.pdf`
- **Word**: `.docx`
- **텍스트**: `.txt`
- **PowerPoint**: `.ppt`, `.pptx`

## 의존성

- `langchain` - LLM 체인 구성
- `langchain-aws` - AWS Bedrock 연동
- `langchain-community` - 문서 로더 및 벡터 저장소
- `faiss-cpu` - 벡터 검색
- `python-pptx` - PPT 파일 처리 (선택사항)

## 향후 계획

1. **추가 문서 형식 지원** (Excel, CSV 등)
2. **다양한 벡터 저장소 지원** (Pinecone, Weaviate 등)
3. **성능 모니터링** 및 메트릭 수집
4. **API 엔드포인트** 추가
5. **배치 처리** 기능 구현

## 기여 방법

1. 이슈 생성
2. 피처 브랜치 생성
3. 코드 작성 및 테스트
4. 풀 리퀘스트 생성

---

이 리팩토링을 통해 코드의 가독성, 유지보수성, 확장성이 크게 향상되었습니다. 각 모듈이 독립적으로 동작하므로 필요에 따라 개별 모듈만 수정하거나 확장할 수 있습니다.
