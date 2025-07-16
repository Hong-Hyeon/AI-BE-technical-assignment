from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

from db.engine import engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()