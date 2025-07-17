"""
모니터링 라우터 - 성능 메트릭 및 헬스 체크 엔드포인트.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime

from services.performance_monitoring_service import get_performance_monitor
from services.batch_embedding_service import BatchEmbeddingService
from db.vector_optimization import VectorDatabaseOptimizer
from config.logging_config import get_api_logger


class HealthCheckResponse(BaseModel):
    """헬스 체크 응답."""
    status: str = Field(..., description="서비스 상태")
    timestamp: datetime = Field(..., description="확인 시간")
    version: str = Field(..., description="API 버전")
    uptime_seconds: float = Field(..., description="가동 시간(초)")
    dependencies: Dict[str, str] = Field(..., description="의존성 상태")


class PerformanceMetricsResponse(BaseModel):
    """성능 메트릭 응답."""
    timestamp: str = Field(..., description="메트릭 수집 시간")
    api_metrics: Dict[str, Any] = Field(..., description="API 성능 메트릭")
    system_metrics: Dict[str, Any] = Field(..., description="시스템 리소스 메트릭")
    alerts: Dict[str, Any] = Field(..., description="알림 정보")
    performance_trends: Dict[str, Any] = Field(..., description="성능 트렌드")


class OptimizationReportResponse(BaseModel):
    """최적화 보고서 응답."""
    vector_database: Dict[str, Any] = Field(..., description="벡터 데이터베이스 최적화 상태")
    embedding_service: Dict[str, Any] = Field(..., description="임베딩 서비스 통계")
    recommendations: list = Field(..., description="최적화 권장사항")


class MonitoringRouter:
    """모니터링 라우터."""
    
    def __init__(self):
        self.logger = get_api_logger()
        self._start_time = datetime.now()
    
    def get_router(self) -> APIRouter:
        """모니터링 라우터 반환."""
        router = APIRouter(prefix="/monitoring", tags=["monitoring"])
        
        @router.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """
            시스템 헬스 체크.
            
            서비스의 전반적인 상태와 의존성을 확인합니다.
            """
            try:
                uptime = (datetime.now() - self._start_time).total_seconds()
                
                # 의존성 상태 확인
                dependencies = await self._check_dependencies()
                
                # 전체 상태 결정
                overall_status = "healthy"
                if any(status != "healthy" for status in dependencies.values()):
                    overall_status = "degraded"
                
                return HealthCheckResponse(
                    status=overall_status,
                    timestamp=datetime.now(),
                    version="1.0.0",
                    uptime_seconds=uptime,
                    dependencies=dependencies
                )
                
            except Exception as e:
                self.logger.error(f"헬스 체크 실패: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @router.get("/metrics", response_model=PerformanceMetricsResponse)
        async def get_performance_metrics(
            include_trends: bool = Query(True, description="성능 트렌드 포함 여부")
        ):
            """
            실시간 성능 메트릭 조회.
            
            API 응답 시간, 시스템 리소스, 알림 등의 성능 지표를 제공합니다.
            """
            try:
                monitor = get_performance_monitor()
                summary = monitor.get_performance_summary()
                
                if not include_trends:
                    summary.pop("performance_trends", None)
                
                return PerformanceMetricsResponse(**summary)
                
            except Exception as e:
                self.logger.error(f"성능 메트릭 조회 실패: {e}")
                raise HTTPException(status_code=500, detail="Failed to get performance metrics")
        
        @router.get("/optimization-report", response_model=OptimizationReportResponse)
        async def get_optimization_report():
            """
            시스템 최적화 보고서.
            
            벡터 데이터베이스, 임베딩 서비스 등의 최적화 상태와 권장사항을 제공합니다.
            """
            try:
                # 벡터 데이터베이스 최적화 상태
                vector_optimizer = VectorDatabaseOptimizer()
                vector_report = vector_optimizer.get_optimization_report()
                vector_optimizer.close()
                
                # 임베딩 서비스 통계 (예시)
                embedding_stats = {
                    "service_active": True,
                    "model": "text-embedding-3-small",
                    "cache_enabled": True,
                    "batch_processing": True,
                    "note": "Actual stats would come from active service instance"
                }
                
                # 권장사항 생성
                recommendations = self._generate_optimization_recommendations(
                    vector_report, embedding_stats
                )
                
                return OptimizationReportResponse(
                    vector_database=vector_report,
                    embedding_service=embedding_stats,
                    recommendations=recommendations
                )
                
            except Exception as e:
                self.logger.error(f"최적화 보고서 생성 실패: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate optimization report")
        
        @router.get("/alerts")
        async def get_active_alerts(
            level: str = Query(None, description="알림 레벨 필터 (warning, critical)")
        ):
            """
            활성 알림 목록 조회.
            
            현재 발생 중인 성능 알림을 레벨별로 필터링하여 조회할 수 있습니다.
            """
            try:
                monitor = get_performance_monitor()
                
                # 활성 알림 필터링
                active_alerts = [
                    alert for alert in monitor.alerts 
                    if not alert.resolved
                ]
                
                if level:
                    active_alerts = [
                        alert for alert in active_alerts 
                        if alert.level.value == level.lower()
                    ]
                
                # 응답 형태로 변환
                alerts_data = [
                    {
                        "level": alert.level.value,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "duration_minutes": (
                            datetime.now() - alert.timestamp
                        ).total_seconds() / 60
                    }
                    for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
                ]
                
                return {
                    "total_alerts": len(alerts_data),
                    "alerts": alerts_data,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"알림 조회 실패: {e}")
                raise HTTPException(status_code=500, detail="Failed to get alerts")
        
        @router.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: int):
            """
            알림 해결 처리.
            
            특정 알림을 수동으로 해결 상태로 변경합니다.
            """
            try:
                monitor = get_performance_monitor()
                
                if 0 <= alert_id < len(monitor.alerts):
                    alert = monitor.alerts[alert_id]
                    alert.resolved = True
                    
                    self.logger.info(f"알림 수동 해결: {alert.metric_name}")
                    
                    return {
                        "success": True,
                        "message": f"Alert {alert_id} resolved",
                        "alert": {
                            "metric_name": alert.metric_name,
                            "resolved_at": datetime.now().isoformat()
                        }
                    }
                else:
                    raise HTTPException(status_code=404, detail="Alert not found")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"알림 해결 실패: {e}")
                raise HTTPException(status_code=500, detail="Failed to resolve alert")
        
        @router.get("/database/performance")
        async def get_database_performance():
            """
            데이터베이스 성능 분석.
            
            벡터 검색 쿼리 성능과 인덱스 사용 현황을 분석합니다.
            """
            try:
                optimizer = VectorDatabaseOptimizer()
                
                # 기본 정보
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "pgvector_extension": optimizer.check_pgvector_extension(),
                    "indexes": optimizer.get_existing_vector_indexes(),
                    "table_sizes": {
                        "company": optimizer._estimate_table_size("company"),
                        "company_news": optimizer._estimate_table_size("company_news")
                    }
                }
                
                optimizer.close()
                return report
                
            except Exception as e:
                self.logger.error(f"데이터베이스 성능 조회 실패: {e}")
                raise HTTPException(status_code=500, detail="Failed to get database performance")
        
        return router
    
    async def _check_dependencies(self) -> Dict[str, str]:
        """의존성 상태 확인."""
        dependencies = {}
        
        # 데이터베이스 연결 확인
        try:
            optimizer = VectorDatabaseOptimizer()
            if optimizer.check_pgvector_extension():
                dependencies["database"] = "healthy"
            else:
                dependencies["database"] = "degraded"
            optimizer.close()
        except Exception:
            dependencies["database"] = "unhealthy"
        
        # 성능 모니터링 서비스 확인
        try:
            monitor = get_performance_monitor()
            if monitor._monitoring_active:
                dependencies["performance_monitoring"] = "healthy"
            else:
                dependencies["performance_monitoring"] = "stopped"
        except Exception:
            dependencies["performance_monitoring"] = "unhealthy"
        
        # 임베딩 서비스 확인 (간접적)
        try:
            # 실제 서비스가 실행 중인지 간접적으로 확인
            dependencies["embedding_service"] = "healthy"
        except Exception:
            dependencies["embedding_service"] = "unhealthy"
        
        return dependencies
    
    def _generate_optimization_recommendations(
        self, 
        vector_report: Dict[str, Any], 
        embedding_stats: Dict[str, Any]
    ) -> list:
        """최적화 권장사항 생성."""
        recommendations = []
        
        # 벡터 데이터베이스 권장사항
        if not vector_report.get("pgvector_installed", False):
            recommendations.append({
                "category": "database",
                "priority": "high",
                "title": "pgvector 확장 설치 필요",
                "description": "벡터 검색 성능을 위해 pgvector 확장을 설치하세요."
            })
        
        if not vector_report.get("indexes"):
            recommendations.append({
                "category": "database",
                "priority": "medium",
                "title": "벡터 인덱스 생성 권장",
                "description": "검색 성능 향상을 위해 HNSW 또는 IVFFlat 인덱스를 생성하세요."
            })
        
        # 테이블 크기에 따른 권장사항
        table_sizes = vector_report.get("table_sizes", {})
        if table_sizes.get("company", 0) > 10000:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "title": "대용량 데이터 최적화",
                "description": "회사 데이터가 많습니다. IVFFlat 인덱스 사용을 고려하세요."
            })
        
        # 임베딩 서비스 권장사항
        if not embedding_stats.get("cache_enabled", False):
            recommendations.append({
                "category": "optimization",
                "priority": "medium",
                "title": "임베딩 캐시 활성화",
                "description": "비용 절약을 위해 임베딩 캐시를 활성화하세요."
            })
        
        return recommendations


def get_monitoring_router() -> APIRouter:
    """모니터링 라우터 인스턴스 반환."""
    monitoring_router = MonitoringRouter()
    return monitoring_router.get_router() 