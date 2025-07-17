"""
PostgreSQL pgvector 성능 최적화 모듈.

이 모듈은 다음 기능을 제공합니다:
- 벡터 인덱스 최적화 (IVFFlat, HNSW)
- 쿼리 성능 분석 및 모니터링
- 벡터 검색 최적화 설정
- 동적 인덱스 관리
"""
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from enum import Enum
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from config.logging_config import get_api_logger

load_dotenv()


class IndexType(str, Enum):
    """벡터 인덱스 타입."""
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"
    BRUTE_FORCE = "brute_force"  # 인덱스 없음


class DistanceMetric(str, Enum):
    """거리 측정 방식."""
    COSINE = "vector_cosine_ops"
    L2 = "vector_l2_ops"
    INNER_PRODUCT = "vector_ip_ops"


@dataclass
class IndexConfiguration:
    """인덱스 설정."""
    index_type: IndexType
    distance_metric: DistanceMetric
    lists: Optional[int] = None  # IVFFlat용
    m: Optional[int] = None      # HNSW용
    ef_construction: Optional[int] = None  # HNSW용


@dataclass
class QueryPerformanceMetrics:
    """쿼리 성능 메트릭."""
    query_time_ms: float
    rows_examined: int
    rows_returned: int
    index_used: bool
    index_name: Optional[str]
    query_plan: str


class VectorDatabaseOptimizer:
    """벡터 데이터베이스 최적화 관리자."""
    
    def __init__(self, database_url: Optional[str] = None):
        """초기화."""
        self.logger = get_api_logger()
        self.database_url = database_url or os.getenv("DATABASE_URL")
        
        if not self.database_url:
            raise ValueError("DATABASE_URL이 설정되지 않았습니다.")
        
        self.engine = create_engine(self.database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # 성능 메트릭
        self.query_stats: Dict[str, List[QueryPerformanceMetrics]] = {}
    
    def check_pgvector_extension(self) -> bool:
        """pgvector 확장이 설치되어 있는지 확인."""
        try:
            result = self.session.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            
            is_installed = result is not None
            self.logger.info(f"pgvector 확장 상태: {'설치됨' if is_installed else '미설치'}")
            return is_installed
            
        except Exception as e:
            self.logger.error(f"pgvector 확장 확인 실패: {e}")
            return False
    
    def install_pgvector_extension(self) -> bool:
        """pgvector 확장 설치."""
        try:
            self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.session.commit()
            self.logger.info("✅ pgvector 확장이 설치되었습니다.")
            return True
            
        except Exception as e:
            self.logger.error(f"pgvector 확장 설치 실패: {e}")
            self.session.rollback()
            return False
    
    def get_existing_vector_indexes(self) -> List[Dict[str, Any]]:
        """기존 벡터 인덱스 목록 조회."""
        try:
            result = self.session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE indexdef LIKE '%vector%'
                   OR indexdef LIKE '%ivfflat%'
                   OR indexdef LIKE '%hnsw%'
                ORDER BY schemaname, tablename, indexname
            """)).fetchall()
            
            indexes = [
                {
                    "schema": row.schemaname,
                    "table": row.tablename,
                    "index_name": row.indexname,
                    "definition": row.indexdef
                }
                for row in result
            ]
            
            self.logger.info(f"기존 벡터 인덱스 {len(indexes)}개 발견")
            return indexes
            
        except Exception as e:
            self.logger.error(f"벡터 인덱스 조회 실패: {e}")
            return []
    
    def create_optimized_indexes(
        self, 
        force_recreate: bool = False
    ) -> Dict[str, bool]:
        """최적화된 벡터 인덱스 생성."""
        results = {}
        
        # Company 테이블 인덱스
        company_result = self._create_table_index(
            table="company",
            column="embedding",
            config=IndexConfiguration(
                index_type=IndexType.HNSW,
                distance_metric=DistanceMetric.COSINE,
                m=16,
                ef_construction=64
            ),
            force_recreate=force_recreate
        )
        results["company_embedding_idx"] = company_result
        
        # Company News 테이블 인덱스
        news_result = self._create_table_index(
            table="company_news",
            column="embedding",
            config=IndexConfiguration(
                index_type=IndexType.HNSW,
                distance_metric=DistanceMetric.COSINE,
                m=16,
                ef_construction=64
            ),
            force_recreate=force_recreate
        )
        results["company_news_embedding_idx"] = news_result
        
        # IVFFlat 대안 인덱스 (큰 데이터셋용)
        if self._estimate_table_size("company") > 10000:
            company_ivf_result = self._create_table_index(
                table="company",
                column="embedding",
                config=IndexConfiguration(
                    index_type=IndexType.IVFFLAT,
                    distance_metric=DistanceMetric.COSINE,
                    lists=100
                ),
                index_suffix="ivf",
                force_recreate=force_recreate
            )
            results["company_embedding_ivf_idx"] = company_ivf_result
        
        return results
    
    def _create_table_index(
        self,
        table: str,
        column: str,
        config: IndexConfiguration,
        index_suffix: str = "",
        force_recreate: bool = False
    ) -> bool:
        """테이블에 벡터 인덱스 생성."""
        suffix = f"_{index_suffix}" if index_suffix else ""
        index_name = f"{table}_{column}_{config.index_type.value}{suffix}_idx"
        
        try:
            # 기존 인덱스 확인
            if not force_recreate:
                existing = self.session.execute(text("""
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = :index_name
                """), {"index_name": index_name}).fetchone()
                
                if existing:
                    self.logger.info(f"인덱스 {index_name}이 이미 존재합니다.")
                    return True
            
            # 기존 인덱스 삭제 (재생성 시)
            if force_recreate:
                self.session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                self.logger.info(f"기존 인덱스 {index_name} 삭제")
            
            # 인덱스 생성 쿼리 구성
            if config.index_type == IndexType.HNSW:
                create_sql = f"""
                    CREATE INDEX {index_name} 
                    ON {table} 
                    USING hnsw ({column} {config.distance_metric.value})
                    WITH (m = {config.m}, ef_construction = {config.ef_construction})
                """
            elif config.index_type == IndexType.IVFFLAT:
                create_sql = f"""
                    CREATE INDEX {index_name} 
                    ON {table} 
                    USING ivfflat ({column} {config.distance_metric.value})
                    WITH (lists = {config.lists})
                """
            else:
                self.logger.warning(f"지원되지 않는 인덱스 타입: {config.index_type}")
                return False
            
            # 인덱스 생성 실행
            start_time = time.time()
            self.logger.info(f"인덱스 {index_name} 생성 시작...")
            
            self.session.execute(text(create_sql))
            self.session.commit()
            
            creation_time = time.time() - start_time
            self.logger.info(
                f"✅ 인덱스 {index_name} 생성 완료 ({creation_time:.2f}초)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"인덱스 {index_name} 생성 실패: {e}")
            self.session.rollback()
            return False
    
    def _estimate_table_size(self, table: str) -> int:
        """테이블 크기 추정."""
        try:
            result = self.session.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()
            return result or 0
        except Exception:
            return 0
    
    def analyze_query_performance(
        self,
        query: str,
        params: Dict[str, Any] = None
    ) -> QueryPerformanceMetrics:
        """쿼리 성능 분석."""
        try:
            # EXPLAIN ANALYZE 실행
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            
            result = self.session.execute(text(explain_query), params or {})
            
            explain_result = result.fetchone()[0][0]
            
            # 메트릭 추출
            execution_time = explain_result.get("Execution Time", 0)
            planning_time = explain_result.get("Planning Time", 0)
            total_time = execution_time + planning_time
            
            plan = explain_result.get("Plan", {})
            rows_examined = self._extract_rows_examined(plan)
            rows_returned = plan.get("Actual Rows", 0)
            index_info = self._extract_index_info(plan)
            
            metrics = QueryPerformanceMetrics(
                query_time_ms=total_time,
                rows_examined=rows_examined,
                rows_returned=rows_returned,
                index_used=index_info["used"],
                index_name=index_info["name"],
                query_plan=str(explain_result)
            )
            
            self.logger.debug(
                f"쿼리 성능 분석 완료 | "
                f"시간: {total_time:.2f}ms | "
                f"인덱스 사용: {index_info['used']} | "
                f"검토 행: {rows_examined} | 반환 행: {rows_returned}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"쿼리 성능 분석 실패: {e}")
            return QueryPerformanceMetrics(
                query_time_ms=0,
                rows_examined=0,
                rows_returned=0,
                index_used=False,
                index_name=None,
                query_plan=str(e)
            )
    
    def _extract_rows_examined(self, plan: Dict) -> int:
        """실행 계획에서 검토된 행 수 추출."""
        rows = plan.get("Actual Rows", 0)
        
        # 하위 노드 검사
        if "Plans" in plan:
            for subplan in plan["Plans"]:
                rows += self._extract_rows_examined(subplan)
        
        return rows
    
    def _extract_index_info(self, plan: Dict) -> Dict[str, Any]:
        """실행 계획에서 인덱스 정보 추출."""
        node_type = plan.get("Node Type", "")
        
        if "Index" in node_type:
            return {
                "used": True,
                "name": plan.get("Index Name"),
                "type": node_type
            }
        
        # 하위 노드 검사
        if "Plans" in plan:
            for subplan in plan["Plans"]:
                index_info = self._extract_index_info(subplan)
                if index_info["used"]:
                    return index_info
        
        return {"used": False, "name": None, "type": None}
    
    def benchmark_vector_queries(
        self,
        sample_embeddings: List[List[float]],
        table: str = "company",
        column: str = "embedding"
    ) -> Dict[str, Any]:
        """벡터 쿼리 벤치마크."""
        results = {
            "total_queries": len(sample_embeddings),
            "avg_query_time": 0,
            "median_query_time": 0,
            "index_usage_rate": 0,
            "query_details": []
        }
        
        if not sample_embeddings:
            return results
        
        self.logger.info(f"벡터 쿼리 벤치마크 시작 ({len(sample_embeddings)}개 쿼리)")
        
        query_times = []
        index_used_count = 0
        
        for i, embedding in enumerate(sample_embeddings):
            try:
                query = f"""
                    SELECT id, 1 - ({column} <=> CAST(:embedding AS vector)) as similarity
                    FROM {table}
                    WHERE {column} IS NOT NULL
                    ORDER BY {column} <=> CAST(:embedding AS vector)
                    LIMIT 5
                """
                
                metrics = self.analyze_query_performance(
                    query, {"embedding": embedding}
                )
                
                query_times.append(metrics.query_time_ms)
                if metrics.index_used:
                    index_used_count += 1
                
                results["query_details"].append({
                    "query_id": i,
                    "time_ms": metrics.query_time_ms,
                    "index_used": metrics.index_used,
                    "index_name": metrics.index_name,
                    "rows_examined": metrics.rows_examined
                })
                
            except Exception as e:
                self.logger.error(f"벤치마크 쿼리 {i} 실패: {e}")
        
        if query_times:
            results["avg_query_time"] = sum(query_times) / len(query_times)
            results["median_query_time"] = sorted(query_times)[len(query_times) // 2]
            results["index_usage_rate"] = (index_used_count / len(query_times)) * 100
        
        self.logger.info(
            f"✅ 벤치마크 완료 | "
            f"평균 시간: {results['avg_query_time']:.2f}ms | "
            f"인덱스 사용률: {results['index_usage_rate']:.1f}%"
        )
        
        return results
    
    def optimize_database_settings(self) -> Dict[str, bool]:
        """벡터 검색에 최적화된 데이터베이스 설정."""
        optimizations = {}
        
        # 벡터 검색 관련 설정들
        settings = {
            # 메모리 설정
            "shared_buffers": "256MB",  # 작은 환경에 맞춤
            "effective_cache_size": "1GB",
            
            # 벡터 관련 설정
            "hnsw.ef_search": "40",  # HNSW 검색 정확도
            "ivfflat.probes": "10",  # IVFFlat 검색 범위
            
            # 일반 성능 설정
            "random_page_cost": "1.1",  # SSD 최적화
            "work_mem": "16MB"
        }
        
        for setting, value in settings.items():
            try:
                # 현재 값 확인
                current = self.session.execute(
                    text(f"SHOW {setting}")
                ).scalar()
                
                if current != value:
                    # 설정 변경 시도 (권한이 있는 경우만)
                    try:
                        self.session.execute(
                            text(f"SET {setting} = :value"),
                            {"value": value}
                        )
                        optimizations[setting] = True
                        self.logger.info(f"설정 {setting} = {value} 적용")
                    except Exception:
                        # 세션 레벨에서만 변경 시도
                        optimizations[setting] = False
                        self.logger.warning(f"설정 {setting} 변경 권한 없음")
                else:
                    optimizations[setting] = True
                    
            except Exception as e:
                optimizations[setting] = False
                self.logger.warning(f"설정 {setting} 확인 실패: {e}")
        
        return optimizations
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """최적화 상태 보고서."""
        report = {
            "timestamp": time.time(),
            "pgvector_installed": self.check_pgvector_extension(),
            "indexes": self.get_existing_vector_indexes(),
            "database_settings": {},
            "recommendations": []
        }
        
        # 테이블 크기 확인
        company_size = self._estimate_table_size("company")
        news_size = self._estimate_table_size("company_news")
        
        report["table_sizes"] = {
            "company": company_size,
            "company_news": news_size
        }
        
        # 권장사항 생성
        if company_size > 1000 and not any("hnsw" in idx["definition"].lower() for idx in report["indexes"]):
            report["recommendations"].append("Company 테이블에 HNSW 인덱스 생성 권장")
        
        if news_size > 5000 and not any("company_news" in idx["table"] for idx in report["indexes"]):
            report["recommendations"].append("Company News 테이블에 벡터 인덱스 생성 권장")
        
        if company_size > 50000:
            report["recommendations"].append("대용량 데이터셋: IVFFlat 인덱스 고려")
        
        return report
    
    def close(self):
        """리소스 정리."""
        if self.session:
            self.session.close()


# 편의 함수들
async def setup_vector_optimization() -> VectorDatabaseOptimizer:
    """벡터 데이터베이스 최적화 설정."""
    optimizer = VectorDatabaseOptimizer()
    
    # 기본 설정
    if not optimizer.check_pgvector_extension():
        optimizer.install_pgvector_extension()
    
    # 인덱스 생성
    optimizer.create_optimized_indexes()
    
    # 데이터베이스 설정 최적화
    optimizer.optimize_database_settings()
    
    return optimizer


def get_optimization_recommendations(
    company_count: int,
    news_count: int
) -> List[str]:
    """데이터 크기에 따른 최적화 권장사항."""
    recommendations = []
    
    if company_count < 1000:
        recommendations.append("소규모 데이터: 브루트 포스 검색으로 충분")
    elif company_count < 10000:
        recommendations.append("중간 규모: HNSW 인덱스 권장")
    else:
        recommendations.append("대규모 데이터: IVFFlat 인덱스 고려")
    
    if news_count > company_count * 10:
        recommendations.append("뉴스 데이터가 많음: 별도 최적화 필요")
    
    return recommendations 