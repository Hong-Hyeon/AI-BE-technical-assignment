from db.session import SessionLocal
from core.logging_config import get_logger

# 로거 생성
logger = get_logger(__name__)


# Dependency
def get_db():
    logger.debug("데이터베이스 세션 생성")
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"데이터베이스 세션 중 오류 발생: {str(e)}")
        db.rollback()
        raise
    finally:
        logger.debug("데이터베이스 세션 종료")
        db.close()