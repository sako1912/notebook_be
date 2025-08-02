from typing import Dict, Any, List, Optional
import logging

from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    검색 전용 서비스
    - 문서 검색 로직
    - 검색 결과 정리
    - 하이브리드 검색 지원
    """
    
    def __init__(self, vector_store_service: VectorStoreService):
        """검색 서비스 초기화"""
        logger.info("검색 서비스 초기화 시작...")
        
        # 벡터 저장소 서비스 주입
        self.vector_store_service = vector_store_service
        
        logger.info("검색 서비스 초기화 완료")
    
    def search(self, query: str, k: int = 4, search_type: str = "vector", 
               filter_dict: Optional[Dict] = None, include_scores: bool = False) -> Dict[str, Any]:
        """
        통합 검색 메서드
        
        Args:
            query: 검색 질의
            k: 반환할 문서 수
            search_type: 검색 유형 ("vector", "hybrid", "keyword")
            filter_dict: 메타데이터 필터
            include_scores: 점수 포함 여부
            
        Returns:
            검색 결과
        """
        try:
            if search_type == "vector":
                return self._vector_search(query, k, filter_dict, include_scores)
            elif search_type == "hybrid":
                return self._hybrid_search(query, k, filter_dict, include_scores)
            elif search_type == "keyword":
                return self._keyword_search(query, k, filter_dict, include_scores)
            else:
                return {
                    "status": "error",
                    "error": f"지원하지 않는 검색 유형: {search_type}"
                }
                
        except Exception as e:
            error_msg = f"검색 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "query": query
            }
    
    def _vector_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None, 
                      include_scores: bool = False) -> Dict[str, Any]:
        """
        벡터 기반 의미 검색
        
        Args:
            query: 검색 질의
            k: 반환할 문서 수
            filter_dict: 메타데이터 필터
            include_scores: 점수 포함 여부
            
        Returns:
            검색 결과
        """
        try:
            if include_scores:
                # 점수와 함께 검색
                search_results = self.vector_store_service.search_with_scores(
                    query=query, k=k, filter_dict=filter_dict
                )
                
                # 결과 정리
                formatted_docs = []
                for doc, score in search_results:
                    formatted_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
            else:
                # 일반 검색
                search_results = self.vector_store_service.search_documents(
                    query=query, k=k, filter_dict=filter_dict
                )
                
                # 결과 정리
                formatted_docs = []
                for doc in search_results:
                    formatted_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    })
            
            if not formatted_docs:
                return {
                    "status": "success",
                    "documents": [],
                    "message": "관련 문서를 찾을 수 없습니다.",
                    "query": query,
                    "search_type": "vector"
                }
            
            logger.info(f"벡터 검색 완료: {len(formatted_docs)}개 문서 발견 - {query[:50]}...")
            
            return {
                "status": "success",
                "documents": formatted_docs,
                "message": f"{len(formatted_docs)}개의 관련 문서를 찾았습니다.",
                "query": query,
                "search_type": "vector"
            }
            
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {str(e)}")
            raise
    
    def _hybrid_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None, 
                      include_scores: bool = False) -> Dict[str, Any]:
        """
        하이브리드 검색 (벡터 + 키워드)
        현재는 벡터 검색만 구현, 향후 키워드 검색 추가 예정
        
        Args:
            query: 검색 질의
            k: 반환할 문서 수
            filter_dict: 메타데이터 필터
            include_scores: 점수 포함 여부
            
        Returns:
            검색 결과
        """
        try:
            # 현재는 벡터 검색만 사용
            # 향후 키워드 검색과 결합하여 하이브리드 검색 구현
            vector_results = self._vector_search(query, k, filter_dict, include_scores)
            
            # 하이브리드 검색 표시
            if vector_results["status"] == "success":
                vector_results["search_type"] = "hybrid"
                vector_results["message"] = vector_results["message"].replace("관련 문서", "하이브리드 검색으로 관련 문서")
            
            return vector_results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 중 오류 발생: {str(e)}")
            raise
    
    def _keyword_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None, 
                       include_scores: bool = False) -> Dict[str, Any]:
        """
        키워드 기반 검색
        현재는 메타데이터 내 키워드 검색으로 구현
        
        Args:
            query: 검색 질의
            k: 반환할 문서 수
            filter_dict: 메타데이터 필터
            include_scores: 점수 포함 여부
            
        Returns:
            검색 결과
        """
        try:
            # 모든 문서를 가져와서 키워드 매칭
            # 실제로는 전용 키워드 검색 인덱스 필요
            all_docs = self.vector_store_service.search_documents(
                query="", k=1000  # 많은 문서를 가져와서 필터링
            )
            
            # 키워드 매칭
            keyword_matches = []
            query_lower = query.lower()
            
            for doc in all_docs:
                content_lower = doc.page_content.lower()
                if query_lower in content_lower:
                    # 키워드 매칭 점수 계산 (단순히 등장 횟수)
                    score = content_lower.count(query_lower)
                    keyword_matches.append((doc, score))
            
            # 점수 순으로 정렬하고 상위 k개 선택
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = keyword_matches[:k]
            
            # 결과 정리
            formatted_docs = []
            for doc, score in top_matches:
                formatted_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score if include_scores else None
                })
            
            if not formatted_docs:
                return {
                    "status": "success",
                    "documents": [],
                    "message": "키워드에 해당하는 문서를 찾을 수 없습니다.",
                    "query": query,
                    "search_type": "keyword"
                }
            
            logger.info(f"키워드 검색 완료: {len(formatted_docs)}개 문서 발견 - {query[:50]}...")
            
            return {
                "status": "success",
                "documents": formatted_docs,
                "message": f"{len(formatted_docs)}개의 키워드 매칭 문서를 찾았습니다.",
                "query": query,
                "search_type": "keyword"
            }
            
        except Exception as e:
            logger.error(f"키워드 검색 중 오류 발생: {str(e)}")
            raise
    
    def multi_query_search(self, queries: List[str], k: int = 4, 
                          search_type: str = "vector") -> Dict[str, Any]:
        """
        다중 쿼리 검색
        
        Args:
            queries: 검색 질의 리스트
            k: 각 쿼리당 반환할 문서 수
            search_type: 검색 유형
            
        Returns:
            통합 검색 결과
        """
        try:
            all_results = []
            seen_docs = set()  # 중복 제거용
            
            for query in queries:
                result = self.search(query, k, search_type, include_scores=True)
                
                if result["status"] == "success":
                    for doc_info in result["documents"]:
                        # 문서 내용으로 중복 체크 (더 정교한 방법 필요시 해시 사용)
                        doc_key = doc_info["content"][:100]  # 처음 100자로 구분
                        
                        if doc_key not in seen_docs:
                            seen_docs.add(doc_key)
                            doc_info["source_query"] = query
                            all_results.append(doc_info)
            
            # 점수 순으로 정렬 (점수가 있는 경우)
            if all_results and all_results[0]["score"] is not None:
                all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 상위 k개만 선택
            final_results = all_results[:k]
            
            logger.info(f"다중 쿼리 검색 완료: {len(queries)}개 쿼리 -> {len(final_results)}개 문서")
            
            return {
                "status": "success",
                "documents": final_results,
                "message": f"{len(queries)}개 쿼리로 {len(final_results)}개의 문서를 찾았습니다.",
                "queries": queries,
                "search_type": f"multi_{search_type}"
            }
            
        except Exception as e:
            error_msg = f"다중 쿼리 검색 중 오류 발생: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "queries": queries
            }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        검색 통계 정보 반환
        
        Returns:
            검색 통계
        """
        try:
            vector_store_status = self.vector_store_service.get_status()
            
            return {
                "status": "success",
                "total_documents": vector_store_status.get("document_count", 0),
                "embedding_dimension": vector_store_status.get("embedding_dimension", 0),
                "supported_search_types": ["vector", "hybrid", "keyword"],
                "message": "검색 서비스가 정상 작동 중입니다."
            }
            
        except Exception as e:
            error_msg = f"검색 통계 조회 실패: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            } 