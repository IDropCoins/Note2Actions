import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    zvec_db_path: str = "./zvec_db"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 150
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
