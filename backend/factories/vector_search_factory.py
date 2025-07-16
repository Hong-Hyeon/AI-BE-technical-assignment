"""
벡터 검색 팩토리 - PostgreSQL pgvector를 활용한 의미적 검색 구현
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from models.company import Company, CompanyNews

load_dotenv()


class VectorSearchEngine:
    """pgvector 기반 벡터 검색 엔진"""
    
    def __init__(self):
        """초기화"""
        # OpenAI 임베딩 모델
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
        """유사한 회사 검색"""
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
            print(f"회사 벡터 검색 오류: {e}")
            return []
    
    def search_similar_news(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.3,
        company_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """유사한 뉴스 검색"""
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
                    "news_date": row.news_date.isoformat() if row.news_date else None,
                    "original_link": row.original_link,
                    "company_name": row.company_name,
                    "company_id": row.company_id,
                    "similarity": float(row.similarity)
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"뉴스 벡터 검색 오류: {e}")
            return []
    
    def search_talent_related_context(
        self, 
        talent_data: Dict[str, Any],
        company_limit: int = 3,
        news_limit: int = 5
    ) -> Dict[str, Any]:
        """인재 데이터 기반 관련 컨텍스트 검색"""
        context = {
            "companies": [],
            "news": [],
            "search_queries": []
        }
        
        try:
            # 교육 기반 검색 쿼리
            education_queries = []
            for edu in talent_data.get("educations", []):
                query = f"{edu.get('school_name', '')} {edu.get('field_of_study', '')} 졸업생 취업 동향"
                education_queries.append(query)
            
            # 경력 기반 검색 쿼리
            position_queries = []
            for pos in talent_data.get("positions", []):
                query = f"{pos.get('company_name', '')} {pos.get('title', '')} 직무 경험"
                position_queries.append(query)
            
            # 스킬 기반 검색 쿼리
            skills = talent_data.get("skills", [])
            if skills:
                skill_query = f"{' '.join(skills[:5])} 기술 전문가 회사"
                position_queries.append(skill_query)
            
            all_queries = education_queries + position_queries
            context["search_queries"] = all_queries
            
            # 각 쿼리로 검색 실행
            all_companies = {}
            all_news = []
            
            for query in all_queries:
                if not query.strip():
                    continue
                    
                # 회사 검색
                companies = self.search_similar_companies(
                    query, 
                    limit=company_limit,
                    similarity_threshold=0.25
                )
                for company in companies:
                    company_id = company["id"]
                    if company_id not in all_companies:
                        all_companies[company_id] = company
                    else:
                        # 유사도가 더 높은 경우 업데이트
                        if company["similarity"] > all_companies[company_id]["similarity"]:
                            all_companies[company_id] = company
                
                # 뉴스 검색
                news = self.search_similar_news(
                    query,
                    limit=news_limit,
                    similarity_threshold=0.25
                )
                all_news.extend(news)
            
            # 결과 정리
            context["companies"] = list(all_companies.values())
            
            # 뉴스는 유사도 순으로 정렬하고 중복 제거
            seen_news = set()
            unique_news = []
            for news in sorted(all_news, key=lambda x: x["similarity"], reverse=True):
                news_key = (news["title"], news["company_id"])
                if news_key not in seen_news:
                    seen_news.add(news_key)
                    unique_news.append(news)
                    if len(unique_news) >= news_limit * 2:  # 충분한 뉴스 수집
                        break
            
            context["news"] = unique_news[:news_limit * 2]
            
        except Exception as e:
            print(f"인재 관련 컨텍스트 검색 오류: {e}")
        
        return context
    
    def close(self):
        """연결 종료"""
        self.session.close()


class VectorSearchFactory:
    """벡터 검색 팩토리"""
    
    @staticmethod
    def create_search_engine() -> VectorSearchEngine:
        """벡터 검색 엔진 생성"""
        return VectorSearchEngine()


class VectorSearchManager:
    """벡터 검색 관리자"""
    
    def __init__(self):
        self.search_engine = None
    
    def get_search_engine(self) -> VectorSearchEngine:
        """벡터 검색 엔진 인스턴스 가져오기 (Lazy Loading)"""
        if self.search_engine is None:
            self.search_engine = VectorSearchFactory.create_search_engine()
        return self.search_engine
    
    def search_talent_context(self, talent_data: Dict[str, Any]) -> Dict[str, Any]:
        """인재 데이터 기반 컨텍스트 검색"""
        engine = self.get_search_engine()
        return engine.search_talent_related_context(talent_data)
    
    def close(self):
        """리소스 정리"""
        if self.search_engine:
            self.search_engine.close()
            self.search_engine = None 