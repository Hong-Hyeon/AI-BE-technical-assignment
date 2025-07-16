from sqlalchemy import create_engine

from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception:
    raise