from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from clinical_rag.schemas import (
    ChromaConfig,
    ChunkConfig,
    EmbedConfig,
    ParserConfig,
    SmokeQueryConfig,
)

DEFAULT_CHUNK_RATIONALE = (
    "Section-aware packing at ~400 tokens with ~12% overlap matches the Day 1 "
    "start band (300–500 tokens, 10–15% overlap). Headings stay on chunks so "
    "citations can name a section without a later retrofit, and windows stay "
    "small enough for local dense retrieval."
)

DEFAULT_EMBED_RATIONALE = (
    "BAAI/bge-m3 runs locally via sentence-transformers (no API key), is "
    "multilingual enough for mixed English/Arabic corpora later, and the smoke "
    "query uses the same model as the index. Operators can switch to "
    "BAAI/bge-small-en-v1.5 if memory is tight."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    smoke_query: SmokeQueryConfig = SmokeQueryConfig()
    parser: ParserConfig = ParserConfig()
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    chroma: ChromaConfig = ChromaConfig()
    uploads_dir: Path = Path("data/uploads")
    jobs_dir: Path = Path("artifacts/jobs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def make_job_id() -> str:
    return uuid.uuid4().hex[:12]


def make_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()[:40]
    return slug or "doc"


def embed_model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)


def collection_name(corpus_id: str, strategy_id: str, embed_model_id: str) -> str:
    raw = f"{corpus_id}__{strategy_id}__{embed_model_slug(embed_model_id)}"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    if len(name) < 3:
        name = f"col_{name}"
    return name[:63]


def detect_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
