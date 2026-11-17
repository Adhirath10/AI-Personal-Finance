from collections.abc import Generator

from fastapi import HTTPException
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config.settings import settings


def _build_sqlite_url() -> URL:
    return URL.create("sqlite", database=str(settings.resolved_database_path))


engine: Engine = create_engine(
    _build_sqlite_url(),
    connect_args={"check_same_thread": False, "timeout": 10},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for FastAPI dependencies."""
    if not settings.resolved_database_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Configured finance database does not exist.",
        )
    database_session = SessionLocal()
    try:
        yield database_session
    finally:
        database_session.close()


def verify_database_connection() -> None:
    """Open the configured database and execute a harmless read-only query."""
    if not settings.resolved_database_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Configured finance database does not exist.",
        )
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))