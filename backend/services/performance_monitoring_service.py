"""
성능 모니터링 서비스.

이 서비스는 다음 기능을 제공합니다:
- API 응답 시간 추적
- 토큰 사용량 모니터링  
- 벡터 검색 성능 메트릭
- 시스템 리소스 사용량 추적
- 실시간 성능 알림
"""
import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import threading
from contextlib import asynccontextmanager

from config.logging_config import get_api_logger


class MetricType(str, Enum):
    """메트릭 타입."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """알림 레벨."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """성능 메트릭."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class APICallMetrics:
    """API 호출 메트릭."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    talent_id: Optional[str] = None
    llm_model: Optional[str] = None
    tokens_used: int = 0
    embedding_cost: float = 0.0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    client_ip: str = ""
    user_agent: str = ""
    trace_id: str = ""


@dataclass
class SystemMetrics:
    """시스템 메트릭."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """성능 알림."""
    level: AlertLevel
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class PerformanceMonitor:
    """성능 모니터링 서비스."""
    
    def __init__(
        self,
        metrics_retention_hours: int = 24,
        alert_check_interval: int = 60
    ):
        """초기화."""
        self.logger = get_api_logger()
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        self.alert_check_interval = alert_check_interval
        
        # 메트릭 저장소
        self.api_metrics: deque = deque(maxlen=10000)
        self.system_metrics: deque = deque(maxlen=1000)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 알림 관리
        self.alerts: List[Alert] = []
        self.alert_thresholds: Dict[str, Dict] = {
            "api_response_time": {"warning": 2000, "critical": 5000},  # ms
            "cpu_usage": {"warning": 70, "critical": 90},  # %
            "memory_usage": {"warning": 80, "critical": 95},  # %
            "error_rate": {"warning": 5, "critical": 10},  # %
            "token_usage_rate": {"warning": 75, "critical": 90}  # % of budget
        }
        
        # 성능 통계
        self.stats_cache: Dict[str, Any] = {}
        self.stats_cache_ttl = 300  # 5분
        self.last_stats_update = 0
        
        # 백그라운드 작업
        self._monitoring_active = False
        self._monitoring_task = None
        
        # 스레드 안전성
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """모니터링 시작."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._background_monitoring())
            self.logger.info("🔍 성능 모니터링이 시작되었습니다.")
    
    def stop_monitoring(self):
        """모니터링 중지."""
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
        self.logger.info("🔍 성능 모니터링이 중지되었습니다.")
    
    async def _background_monitoring(self):
        """백그라운드 모니터링 작업."""
        while self._monitoring_active:
            try:
                # 시스템 메트릭 수집
                await self._collect_system_metrics()
                
                # 알림 확인
                self._check_alerts()
                
                # 오래된 메트릭 정리
                self._cleanup_old_metrics()
                
                await asyncio.sleep(self.alert_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"백그라운드 모니터링 오류: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집."""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            
            # 네트워크 통계
            network = psutil.net_io_counters()
            
            # 활성 연결 수 (예시)
            active_connections = len(psutil.net_connections(kind='inet'))
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=active_connections
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
            
        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 실패: {e}")
    
    def record_api_call(self, metrics: APICallMetrics):
        """API 호출 메트릭 기록."""
        with self._lock:
            self.api_metrics.append(metrics)
        
        # 느린 API 호출 로깅
        if metrics.response_time_ms > 2000:
            self.logger.warning(
                f"🐌 느린 API 호출 감지 | "
                f"엔드포인트: {metrics.endpoint} | "
                f"응답시간: {metrics.response_time_ms:.2f}ms | "
                f"상태코드: {metrics.status_code}"
            )
    
    def record_custom_metric(self, metric: PerformanceMetric):
        """커스텀 메트릭 기록."""
        with self._lock:
            self.custom_metrics[metric.name].append(metric)
    
    def _check_alerts(self):
        """알림 조건 확인."""
        current_time = datetime.now()
        
        # API 응답 시간 알림
        self._check_api_response_time_alerts(current_time)
        
        # 시스템 리소스 알림
        self._check_system_resource_alerts(current_time)
        
        # 에러율 알림
        self._check_error_rate_alerts(current_time)
    
    def _check_api_response_time_alerts(self, current_time: datetime):
        """API 응답 시간 알림 확인."""
        if not self.api_metrics:
            return
        
        # 최근 5분간의 평균 응답 시간
        recent_cutoff = current_time - timedelta(minutes=5)
        recent_calls = [
            m for m in self.api_metrics 
            if m.timestamp >= recent_cutoff
        ]
        
        if not recent_calls:
            return
        
        avg_response_time = sum(m.response_time_ms for m in recent_calls) / len(recent_calls)
        
        self._check_threshold_alert(
            "api_response_time",
            avg_response_time,
            f"평균 API 응답 시간이 {avg_response_time:.2f}ms입니다.",
            current_time
        )
    
    def _check_system_resource_alerts(self, current_time: datetime):
        """시스템 리소스 알림 확인."""
        if not self.system_metrics:
            return
        
        latest_metrics = self.system_metrics[-1]
        
        # CPU 사용률 확인
        self._check_threshold_alert(
            "cpu_usage",
            latest_metrics.cpu_percent,
            f"CPU 사용률이 {latest_metrics.cpu_percent:.1f}%입니다.",
            current_time
        )
        
        # 메모리 사용률 확인
        self._check_threshold_alert(
            "memory_usage",
            latest_metrics.memory_percent,
            f"메모리 사용률이 {latest_metrics.memory_percent:.1f}%입니다.",
            current_time
        )
    
    def _check_error_rate_alerts(self, current_time: datetime):
        """에러율 알림 확인."""
        if not self.api_metrics:
            return
        
        # 최근 10분간의 에러율
        recent_cutoff = current_time - timedelta(minutes=10)
        recent_calls = [
            m for m in self.api_metrics 
            if m.timestamp >= recent_cutoff
        ]
        
        if len(recent_calls) < 10:  # 충분한 샘플이 없으면 스킵
            return
        
        error_calls = [m for m in recent_calls if m.status_code >= 400]
        error_rate = (len(error_calls) / len(recent_calls)) * 100
        
        self._check_threshold_alert(
            "error_rate",
            error_rate,
            f"API 에러율이 {error_rate:.1f}%입니다.",
            current_time
        )
    
    def _check_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        message: str,
        timestamp: datetime
    ):
        """임계값 기반 알림 확인."""
        thresholds = self.alert_thresholds.get(metric_name, {})
        
        # 기존 미해결 알림 확인
        existing_alert = next(
            (a for a in self.alerts 
             if a.metric_name == metric_name and not a.resolved),
            None
        )
        
        if current_value >= thresholds.get("critical", float('inf')):
            if not existing_alert or existing_alert.level != AlertLevel.CRITICAL:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=thresholds["critical"],
                    message=f"🚨 CRITICAL: {message}",
                    timestamp=timestamp
                )
                self.alerts.append(alert)
                self.logger.critical(alert.message)
                
        elif current_value >= thresholds.get("warning", float('inf')):
            if not existing_alert or existing_alert.level == AlertLevel.INFO:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=thresholds["warning"],
                    message=f"⚠️ WARNING: {message}",
                    timestamp=timestamp
                )
                self.alerts.append(alert)
                self.logger.warning(alert.message)
        
        else:
            # 임계값 이하로 내려간 경우 알림 해결
            if existing_alert:
                existing_alert.resolved = True
                self.logger.info(f"✅ 알림 해결: {metric_name} = {current_value}")
    
    def _cleanup_old_metrics(self):
        """오래된 메트릭 정리."""
        current_time = datetime.now()
        cutoff_time = current_time - self.metrics_retention
        
        with self._lock:
            # API 메트릭 정리
            self.api_metrics = deque(
                [m for m in self.api_metrics if m.timestamp >= cutoff_time],
                maxlen=self.api_metrics.maxlen
            )
            
            # 시스템 메트릭 정리
            self.system_metrics = deque(
                [m for m in self.system_metrics if m.timestamp >= cutoff_time],
                maxlen=self.system_metrics.maxlen
            )
            
            # 커스텀 메트릭 정리
            for name, metrics in self.custom_metrics.items():
                self.custom_metrics[name] = deque(
                    [m for m in metrics if m.timestamp >= cutoff_time],
                    maxlen=metrics.maxlen
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보."""
        current_time = time.time()
        
        # 캐시된 결과 확인
        if (current_time - self.last_stats_update) < self.stats_cache_ttl:
            return self.stats_cache
        
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "api_metrics": self._get_api_metrics_summary(),
                "system_metrics": self._get_system_metrics_summary(),
                "alerts": self._get_alerts_summary(),
                "performance_trends": self._get_performance_trends()
            }
        
        # 캐시 업데이트
        self.stats_cache = summary
        self.last_stats_update = current_time
        
        return summary
    
    def _get_api_metrics_summary(self) -> Dict[str, Any]:
        """API 메트릭 요약."""
        if not self.api_metrics:
            return {"total_calls": 0}
        
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_calls = [m for m in self.api_metrics if m.timestamp >= recent_cutoff]
        
        if not recent_calls:
            return {"total_calls": len(self.api_metrics), "recent_calls": 0}
        
        # 기본 통계
        response_times = [m.response_time_ms for m in recent_calls]
        status_codes = [m.status_code for m in recent_calls]
        
        # 에러율 계산
        error_count = len([code for code in status_codes if code >= 400])
        error_rate = (error_count / len(recent_calls)) * 100 if recent_calls else 0
        
        # 토큰 사용량
        total_tokens = sum(m.tokens_used for m in recent_calls)
        total_cost = sum(m.embedding_cost for m in recent_calls)
        
        # 캐시 히트율
        cache_hits = len([m for m in recent_calls if m.cache_hit])
        cache_hit_rate = (cache_hits / len(recent_calls)) * 100 if recent_calls else 0
        
        return {
            "total_calls": len(self.api_metrics),
            "recent_calls_1h": len(recent_calls),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "error_rate_percent": error_rate,
            "total_tokens_used": total_tokens,
            "total_cost_usd": total_cost,
            "cache_hit_rate_percent": cache_hit_rate
        }
    
    def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """시스템 메트릭 요약."""
        if not self.system_metrics:
            return {}
        
        latest = self.system_metrics[-1]
        
        return {
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "memory_used_mb": latest.memory_used_mb,
            "disk_usage_percent": latest.disk_usage_percent,
            "active_connections": latest.active_connections,
            "last_updated": latest.timestamp.isoformat()
        }
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """알림 요약."""
        active_alerts = [a for a in self.alerts if not a.resolved]
        recent_alerts = [
            a for a in self.alerts 
            if a.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        alert_counts = defaultdict(int)
        for alert in active_alerts:
            alert_counts[alert.level.value] += 1
        
        return {
            "active_alerts": len(active_alerts),
            "recent_alerts_24h": len(recent_alerts),
            "alert_counts": dict(alert_counts),
            "latest_alerts": [
                {
                    "level": a.level.value,
                    "metric": a.metric_name,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 분석."""
        if len(self.api_metrics) < 10:
            return {}
        
        # 시간대별 응답 시간 트렌드
        hours_data = defaultdict(list)
        for metric in self.api_metrics:
            hour = metric.timestamp.hour
            hours_data[hour].append(metric.response_time_ms)
        
        hourly_avg = {
            hour: sum(times) / len(times)
            for hour, times in hours_data.items()
        }
        
        return {
            "hourly_response_times": hourly_avg,
            "trend_direction": self._calculate_trend_direction()
        }
    
    def _calculate_trend_direction(self) -> str:
        """트렌드 방향 계산."""
        if len(self.api_metrics) < 20:
            return "insufficient_data"
        
        # 최근 10개와 이전 10개 비교
        recent = list(self.api_metrics)[-10:]
        previous = list(self.api_metrics)[-20:-10]
        
        recent_avg = sum(m.response_time_ms for m in recent) / len(recent)
        previous_avg = sum(m.response_time_ms for m in previous) / len(previous)
        
        if recent_avg > previous_avg * 1.1:
            return "degrading"
        elif recent_avg < previous_avg * 0.9:
            return "improving"
        else:
            return "stable"


# 컨텍스트 매니저
@asynccontextmanager
async def api_performance_tracking(
    monitor: PerformanceMonitor,
    endpoint: str,
    method: str,
    **kwargs
):
    """API 성능 추적 컨텍스트 매니저."""
    start_time = time.time()
    
    try:
        yield
        # 성공적인 요청
        response_time = (time.time() - start_time) * 1000
        
        metrics = APICallMetrics(
            endpoint=endpoint,
            method=method,
            status_code=200,
            response_time_ms=response_time,
            request_size_bytes=0,  # 필요시 계산
            response_size_bytes=0,  # 필요시 계산
            **kwargs
        )
        
        monitor.record_api_call(metrics)
        
    except Exception:
        # 실패한 요청
        response_time = (time.time() - start_time) * 1000
        
        metrics = APICallMetrics(
            endpoint=endpoint,
            method=method,
            status_code=500,
            response_time_ms=response_time,
            request_size_bytes=0,
            response_size_bytes=0,
            **kwargs
        )
        
        monitor.record_api_call(metrics)
        raise


# 글로벌 모니터 인스턴스
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """글로벌 성능 모니터 인스턴스 가져오기."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def cleanup_performance_monitor():
    """성능 모니터 정리."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None 