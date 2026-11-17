from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "personal_finance.db"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    database_path: Path = DEFAULT_DATABASE_PATH
    finance_user_id: str | None = None
    ai_service_url: str = "http://127.0.0.1:8001"
    ai_service_timeout_seconds: float = 120.0
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_database_path(self) -> Path:
        """Return the configured database path as an absolute path."""
        if self.database_path.is_absolute():
            return self.database_path
        return PROJECT_ROOT / self.database_path

    @property
    def resolved_ai_service_url(self) -> str:
        return self.ai_service_url.rstrip("/")


settings = Settings()