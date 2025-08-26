import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"
MEMORY_DIR = BASE_DIR / "memory"

for d in (DATA_DIR, VECTOR_DIR, MEMORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    temperature: float = 0.2

settings = Settings()

if not settings.openai_api_key:
    print("[WARN] OPENAI_API_KEY not set. Set it before running ingestion or chat.")
