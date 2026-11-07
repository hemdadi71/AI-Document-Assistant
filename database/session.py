
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import Session, sessionmaker


# مسیر اصلی پروژه
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# بارگذاری تنظیمات محلی
# ابتدا .env و سپس تنظیمات Docker بارگذاری می‌شوند.
# متغیرهای از قبل تعریف‌شده در محیط سیستم اولویت دارند.
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env.docker", override=False)


def get_database_url() -> URL:
    """
    Build the PostgreSQL connection URL.

    Production:
        Uses DATABASE_URL.

    Local development:
        Uses POSTGRES_* variables from .env.docker.
    """

    # در محیط Deploy از DATABASE_URL استفاده می‌کنیم.
    database_url = os.getenv("DATABASE_URL", "").strip()

    if database_url:

        # سازگاری با URLهایی که با postgres:// شروع می‌شوند.
        if database_url.startswith("postgres://"):
            database_url = (
                "postgresql://"
                + database_url[len("postgres://"):]
            )

        url = make_url(database_url)

        # استفاده از Psycopg 3 به‌عنوان Driver
        if url.drivername in {"postgres", "postgresql"}:
            url = url.set(drivername="postgresql+psycopg")

        if url.drivername != "postgresql+psycopg":
            raise RuntimeError(
                "DATABASE_URL must use PostgreSQL."
            )

        return url

    # تنظیمات محیط توسعه محلی
    database = os.getenv("POSTGRES_DB")
    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")

    missing = [
        name
        for name, value in {
            "POSTGRES_DB": database,
            "POSTGRES_USER": username,
            "POSTGRES_PASSWORD": password,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing database configuration: "
            + ", ".join(missing)
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=database,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Create and cache the SQLAlchemy Engine.

    Engine manages database connectivity and connection pooling.
    """

    return create_engine(
        get_database_url(),

        # بررسی سلامت اتصال قبل از استفاده مجدد
        pool_pre_ping=True,

        # تنظیمات Connection Pool
        pool_size=5,
        max_overflow=2,
        pool_recycle=1800,

        # حداکثر زمان انتظار برای دریافت اتصال
        pool_timeout=30,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """
    Create and cache the SQLAlchemy Session factory.
    """

    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def get_db() -> Iterator[Session]:
    """
    FastAPI database dependency.

    Provides a database session for each request
    and guarantees that the session is closed.
    """

    db = get_session_factory()()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
