"""Settings: every value comes from the environment with the RAGKB_ prefix, or a .env file.

A test checks that every setting here is documented in .env.example and vice versa.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGKB_", env_file=".env", extra="ignore")

    # storage
    DATA_DIR: Path = Path("./data")

    # server
    HOST: str = "127.0.0.1"
    PORT: int = 8765
    UI_ENABLED: bool = True
    MCP_ENABLED: bool = True
    MCP_PUBLIC: bool = False
    LOG_LEVEL: str = "INFO"

    # retrieval
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER: str = "none"  # none | local
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    TOP_K: int = 6
    MIN_SCORE: float = 0.0
    MAX_CONTEXT_CHARS: int = 12000

    # ingestion
    CHUNK_TOKENS: int = 350
    CHUNK_OVERLAP_TOKENS: int = 60
    MAX_UPLOAD_MB: int = 50

    # answer providers
    PROVIDER: str = "auto"  # auto | claude_code | codex | extractive
    CLAUDE_BIN: str = ""
    CODEX_BIN: str = ""
    CLAUDE_MODEL: str = "sonnet"
    CODEX_MODEL: str = ""
    CODEX_REASONING: str = "medium"
    ANSWER_TIMEOUT_S: int = 180

    # derived paths
    @property
    def lancedb_dir(self) -> Path:
        return self.DATA_DIR / "lancedb"

    @property
    def uploads_dir(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def meta_path(self) -> Path:
        return self.DATA_DIR / "meta.sqlite"

    @property
    def logs_dir(self) -> Path:
        return self.DATA_DIR / "logs"

    def ensure_dirs(self) -> None:
        for p in (self.DATA_DIR, self.lancedb_dir, self.uploads_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
