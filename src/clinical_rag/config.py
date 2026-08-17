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
    PineconeConfig,
    SmokeQueryConfig,
    VectorStoreKind,
    WeaviateConfig,
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
    vector_store: VectorStoreKind = VectorStoreKind.chroma
    chroma: ChromaConfig = ChromaConfig()
    weaviate: WeaviateConfig = WeaviateConfig()
    pinecone: PineconeConfig = PineconeConfig()
    uploads_dir: Path = Path("data/uploads")
    jobs_dir: Path = Path("artifacts/jobs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def store_kwargs(settings: Settings | None = None) -> dict:
    """Connection configs for build_store / smoke query (no secrets logged here)."""
    cfg = settings or get_settings()
    return {"chroma": cfg.chroma, "weaviate": cfg.weaviate, "pinecone": cfg.pinecone}


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


def cuda_total_memory_gb(device_index: int = 0) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024**3)
    except Exception:
        return None


# bge-m3 needs ~2.3 GiB for weights alone; leave headroom for activations and other GPU processes.
_MIN_VRAM_GB_FOR_BGE_M3 = 6.0
_LARGE_EMBED_MODELS = frozenset({"BAAI/bge-m3"})


def resolve_embed_config(cfg: EmbedConfig) -> tuple[EmbedConfig, list[str]]:
    """Pick a model/device/batch_size that fits this machine when device is auto."""
    device = detect_device(cfg.device)
    model_id = cfg.model_id
    batch_size = cfg.batch_size
    warnings: list[str] = []

    vram_gb = cuda_total_memory_gb() if device == "cuda" else None
    if model_id in _LARGE_EMBED_MODELS and vram_gb is not None and vram_gb < _MIN_VRAM_GB_FOR_BGE_M3:
        fallback = cfg.fallback_model_id
        if fallback and fallback != model_id:
            warnings.append(
                f"GPU has {vram_gb:.1f} GiB VRAM; switching embed model from {model_id} "
                f"to {fallback} (bge-m3 needs ~6 GiB on CUDA)."
            )
            model_id = fallback
            batch_size = min(batch_size, 4)
        else:
            warnings.append(
                f"GPU has {vram_gb:.1f} GiB VRAM; moving {model_id} to CPU to avoid CUDA OOM."
            )
            device = "cpu"
            batch_size = min(batch_size, 8)

    if device == "cuda" and vram_gb is not None and vram_gb < 4.5:
        batch_size = min(batch_size, 2)

    resolved = EmbedConfig(
        model_id=model_id,
        fallback_model_id=cfg.fallback_model_id,
        device=device,
        batch_size=batch_size,
        provider=cfg.provider,
    )
    return resolved, warnings
