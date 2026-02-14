import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ZVEC_DB_PATH: str = "./zvec_db"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_MAX_CHARS: int = 1400
    CHUNK_OVERLAP_CHARS: int = 150
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
