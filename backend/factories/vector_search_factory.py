"""
벡터 검색 팩토리 - PostgreSQL pgvector를 활용한 의미적 검색 구현
배치 임베딩 처리로 최적화됨
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from models.company import Company, CompanyNews
from services.batch_embedding_service import BatchEmbeddingService, EmbeddingRequest
from config.logging_config import get_api_logger

load_dotenv()


class VectorSearchEngine:
    """pgvector 기반 벡터 검색 엔진 (배치 최적화)"""
    
    def __init__(self):
        """초기화"""
        self.logger = get_api_logger()
        
        # 배치 임베딩 서비스 초기화
        self.batch_embedding_service = BatchEmbeddingService(
            model="text-embedding-3-small",
            max_batch_size=50,  # 벡터 검색에 최적화된 배치 크기
            cache_enabled=True
        )
        
        # 기존 호환성을 위한 개별 임베딩 모델
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 데이터베이스 연결
        self.engine = create_engine(
            os.getenv("DATABASE_URL")
        )
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def search_similar_companies(
        self, 
        query: str, 
        limit: int = 5, 
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """유사한 회사 검색 (개별 쿼리용)"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings_model.embed_query(query)
            
            # 벡터 검색 실행 (벡터 타입으로 명시적 캐스팅)
            results = self.session.execute(
                text("""
                    SELECT 
                        id,
                        name,
                        data,
                        1 - (embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM company 
                    WHERE embedding IS NOT NULL
                        AND 1 - (embedding <=> CAST(:query_embedding AS vector)) > :similarity_threshold
                    ORDER BY embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :limit
                """),
                {
                    "query_embedding": query_embedding,
                    "similarity_threshold": similarity_threshold,
                    "limit": limit
                }
            ).fetchall()
            
            return [
                {
                    "id": row.id,
                    "name": row.name,
                    "data": row.data,
                    "similarity": float(row.similarity)
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"회사 벡터 검색 오류: {e}")
            return []
    
    def search_similar_news(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.3,
        company_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 뉴스 검색 (개별 쿼리용)"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings_model.embed_query(query)
            
            # 특정 회사들로 필터링하는 경우
            company_filter = ""
            params = {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "limit": limit
            }
            
            if company_ids:
                company_filter = "AND cn.company_id = ANY(:company_ids)"
                params["company_ids"] = company_ids
            
            # 벡터 검색 실행 (벡터 타입으로 명시적 캐스팅)
            results = self.session.execute(
                text(f"""
                    SELECT 
                        cn.id,
                        cn.title,
                        cn.news_date,
                        cn.original_link,
                        c.name as company_name,
                        c.id as company_id,
                        1 - (cn.embedding <=> CAST(:query_embedding AS vector)) as similarity
                    FROM company_news cn
                    JOIN company c ON cn.company_id = c.id
                    WHERE cn.embedding IS NOT NULL
                        AND 1 - (cn.embedding <=> CAST(:query_embedding AS vector)) > :similarity_threshold
                        {company_filter}
                    ORDER BY cn.embedding <=> CAST(:query_embedding AS vector)
                    LIMIT :limit
                """),
                params
            ).fetchall()
            
            return [
                {
                    "id": row.id,
                    "title": row.title,
                    "news_date": row.news_date,
                    "original_link": row.original_link,
                    "company_name": row.company_name,
                    "company_id": row.company_id,
                    "similarity": float(row.similarity)
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"뉴스 벡터 검색 오류: {e}")
            return []
    
    async def batch_search_companies(
        self,
        queries: List[str],
        limit: int = 5,
        similarity_threshold: float = 0.3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        배치로 여러 쿼리에 대한 회사 검색 수행.
        
        Returns:
            Dict mapping query -> search results
        """
        if not queries:
            return {}
        
        self.logger.info(f"🔍 Batch company search for {len(queries)} queries")
        
        try:
            # 배치 임베딩 생성
            embedding_requests = [
                EmbeddingRequest(
                    id=f"company_query_{i}",
                    text=query,
                    priority=2,
                    metadata={"type": "company_search", "index": i}
                )
                for i, query in enumerate(queries)
            ]
            
            embedding_results = await self.batch_embedding_service.get_embeddings_batch(
                embedding_requests, use_cache=True
            )
            
            # 각 임베딩으로 검색 수행
            batch_results = {}
            total_db_queries = 0
            
            for result in embedding_results:
                original_index = int(result.id.split("_")[-1])
                query = queries[original_index]
                
                # 데이터베이스 검색
                db_results = self.session.execute(
                    text("""
                        SELECT 
                            id,
                            name,
                            data,
                            1 - (embedding <=> CAST(:query_embedding AS vector)) as similarity
                        FROM company 
                        WHERE embedding IS NOT NULL
                            AND 1 - (embedding <=> CAST(:query_embedding AS vector)) > :similarity_threshold
                        ORDER BY embedding <=> CAST(:query_embedding AS vector)
                        LIMIT :limit
                    """),
                    {
                        "query_embedding": result.embedding,
                        "similarity_threshold": similarity_threshold,
                        "limit": limit
                    }
                ).fetchall()
                
                batch_results[query] = [
                    {
                        "id": row.id,
                        "name": row.name,
                        "data": row.data,
                        "similarity": float(row.similarity)
                    }
                    for row in db_results
                ]
                total_db_queries += 1
            
            # 배치 처리 통계 로깅
            embedding_stats = self.batch_embedding_service.get_service_stats()
            self.logger.info(
                f"✅ Batch company search completed | "
                f"Queries: {len(queries)} | DB searches: {total_db_queries} | "
                f"Embedding cost: ${embedding_stats['total_cost']:.6f}"
            )
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"배치 회사 검색 오류: {e}")
            return {query: [] for query in queries}
    
    async def batch_search_news(
        self,
        queries: List[str],
        limit: int = 10,
        similarity_threshold: float = 0.3,
        company_ids: Optional[List[int]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        배치로 여러 쿼리에 대한 뉴스 검색 수행.
        
        Returns:
            Dict mapping query -> search results
        """
        if not queries:
            return {}
        
        self.logger.info(f"📰 Batch news search for {len(queries)} queries")
        
        try:
            # 배치 임베딩 생성
            embedding_requests = [
                EmbeddingRequest(
                    id=f"news_query_{i}",
                    text=query,
                    priority=2,
                    metadata={"type": "news_search", "index": i}
                )
                for i, query in enumerate(queries)
            ]
            
            embedding_results = await self.batch_embedding_service.get_embeddings_batch(
                embedding_requests, use_cache=True
            )
            
            # 각 임베딩으로 검색 수행
            batch_results = {}
            total_db_queries = 0
            
            # 회사 필터 설정
            company_filter = ""
            base_params = {
                "similarity_threshold": similarity_threshold,
                "limit": limit
            }
            
            if company_ids:
                company_filter = "AND cn.company_id = ANY(:company_ids)"
                base_params["company_ids"] = company_ids
            
            for result in embedding_results:
                original_index = int(result.id.split("_")[-1])
                query = queries[original_index]
                
                # 파라미터에 임베딩 추가
                params = {**base_params, "query_embedding": result.embedding}
                
                # 데이터베이스 검색
                db_results = self.session.execute(
                    text(f"""
                        SELECT 
                            cn.id,
                            cn.title,
                            cn.news_date,
                            cn.original_link,
                            c.name as company_name,
                            c.id as company_id,
                            1 - (cn.embedding <=> CAST(:query_embedding AS vector)) as similarity
                        FROM company_news cn
                        JOIN company c ON cn.company_id = c.id
                        WHERE cn.embedding IS NOT NULL
                            AND 1 - (cn.embedding <=> CAST(:query_embedding AS vector)) > :similarity_threshold
                            {company_filter}
                        ORDER BY cn.embedding <=> CAST(:query_embedding AS vector)
                        LIMIT :limit
                    """),
                    params
                ).fetchall()
                
                batch_results[query] = [
                    {
                        "id": row.id,
                        "title": row.title,
                        "news_date": row.news_date,
                        "original_link": row.original_link,
                        "company_name": row.company_name,
                        "company_id": row.company_id,
                        "similarity": float(row.similarity)
                    }
                    for row in db_results
                ]
                total_db_queries += 1
            
            # 배치 처리 통계 로깅
            embedding_stats = self.batch_embedding_service.get_service_stats()
            self.logger.info(
                f"✅ Batch news search completed | "
                f"Queries: {len(queries)} | DB searches: {total_db_queries} | "
                f"Embedding cost: ${embedding_stats['total_cost']:.6f}"
            )
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"배치 뉴스 검색 오류: {e}")
            return {query: [] for query in queries}
    
    async def search_talent_related_context(
        self, 
        talent_data: Dict[str, Any],
        company_limit: int = 5, 
        news_limit: int = 10
    ) -> Dict[str, Any]:
        """
        최적화된 배치 처리로 인재 관련 컨텍스트 검색.
        """
        context = {
            "companies": [],
            "news": [],
            "search_queries": [],
            "metadata": {
                "total_queries": 0,
                "embedding_cache_hits": 0,
                "total_cost": 0.0
            }
        }
        
        try:
            # 검색 쿼리 생성
            all_queries = []
            
            # 교육 기반 검색 쿼리
            education_queries = []
            for edu in talent_data.get("educations", []):
                query = f"{edu.get('school_name', '')} {edu.get('field_of_study', '')} 졸업생 취업 동향"
                if query.strip():
                    education_queries.append(query)
            
            # 경력 기반 검색 쿼리
            position_queries = []
            for pos in talent_data.get("positions", []):
                query = f"{pos.get('company_name', '')} {pos.get('title', '')} 직무 경험"
                if query.strip():
                    position_queries.append(query)
            
            # 스킬 기반 검색 쿼리
            skills = talent_data.get("skills", [])
            if skills:
                skill_query = f"{' '.join(skills[:5])} 기술 전문가 회사"
                position_queries.append(skill_query)
            
            all_queries = education_queries + position_queries
            context["search_queries"] = all_queries
            context["metadata"]["total_queries"] = len(all_queries)
            
            if not all_queries:
                self.logger.warning("검색 쿼리가 생성되지 않았습니다.")
                return context
            
            # 배치 검색 수행
            self.logger.info(f"🚀 Starting optimized batch search for {len(all_queries)} queries")
            
            # 회사와 뉴스를 병렬로 검색
            import asyncio
            
            company_results_task = self.batch_search_companies(
                all_queries, limit=company_limit, similarity_threshold=0.25
            )
            news_results_task = self.batch_search_news(
                all_queries, limit=news_limit, similarity_threshold=0.25
            )
            
            company_results, news_results = await asyncio.gather(
                company_results_task, news_results_task
            )
            
            # 결과 통합 및 중복 제거
            all_companies = {}
            all_news = []
            
            for query, companies in company_results.items():
                for company in companies:
                    company_id = company["id"]
                    if company_id not in all_companies:
                        all_companies[company_id] = company
                    else:
                        # 유사도가 더 높은 경우 업데이트
                        if company["similarity"] > all_companies[company_id]["similarity"]:
                            all_companies[company_id] = company
            
            for query, news_list in news_results.items():
                all_news.extend(news_list)
            
            # 뉴스 중복 제거 및 정렬
            seen_news = set()
            unique_news = []
            for news in sorted(all_news, key=lambda x: x["similarity"], reverse=True):
                news_key = (news["title"], news["company_id"])
                if news_key not in seen_news:
                    seen_news.add(news_key)
                    unique_news.append(news)
                    if len(unique_news) >= news_limit * 2:
                        break
            
            context["companies"] = list(all_companies.values())
            context["news"] = unique_news[:news_limit * 2]
            
            # 메타데이터 업데이트
            embedding_stats = self.batch_embedding_service.get_service_stats()
            context["metadata"].update({
                "companies_found": len(context["companies"]),
                "news_found": len(context["news"]),
                "embedding_cache_hits": embedding_stats.get("cache", {}).get("hit_count", 0),
                "total_cost": embedding_stats["total_cost"],
                "cache_hit_rate": embedding_stats.get("cache", {}).get("hit_rate", 0)
            })
            
            self.logger.info(
                f"🎯 Optimized search completed | "
                f"Companies: {len(context['companies'])} | "
                f"News: {len(context['news'])} | "
                f"Cost: ${embedding_stats['total_cost']:.6f} | "
                f"Cache hit rate: {context['metadata']['cache_hit_rate']:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"인재 관련 컨텍스트 검색 오류: {e}")
        
        return context
    
    def close(self):
        """리소스 정리"""
        if self.session:
            self.session.close()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """벡터 검색 최적화 통계"""
        return {
            "embedding_service": self.batch_embedding_service.get_service_stats(),
            "database_connection": "active" if self.session else "closed"
        }


# 기존 팩토리 클래스들 유지 (호환성)
class VectorSearchFactory:
    """벡터 검색 엔진 팩토리"""
    
    @staticmethod
    def create_search_engine() -> VectorSearchEngine:
        """벡터 검색 엔진 생성"""
        return VectorSearchEngine()


class VectorSearchManager:
    """벡터 검색 관리자 (개선됨)"""
    
    def __init__(self):
        self.search_engine = None
    
    def get_search_engine(self) -> VectorSearchEngine:
        """벡터 검색 엔진 인스턴스 가져오기 (Lazy Loading)"""
        if self.search_engine is None:
            self.search_engine = VectorSearchFactory.create_search_engine()
        return self.search_engine
    
    def search_talent_context(self, talent_data: Dict[str, Any]) -> Dict[str, Any]:
        """인재 데이터 기반 컨텍스트 검색 (비동기 래퍼)"""
        import asyncio
        engine = self.get_search_engine()
        
        # 비동기 함수를 동기적으로 실행
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                engine.search_talent_related_context(talent_data)
            )
        except RuntimeError:
            # 새 이벤트 루프 생성
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    engine.search_talent_related_context(talent_data)
                )
            finally:
                loop.close()
    
    def close(self):
        """리소스 정리"""
        if self.search_engine:
            self.search_engine.close()
            self.search_engine = None 