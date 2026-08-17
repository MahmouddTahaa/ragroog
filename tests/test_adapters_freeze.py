from pathlib import Path

import pytest

from clinical_rag.adapters.embedders import build_embedder
from clinical_rag.adapters.stores import build_store, pinecone_index_names, weaviate_collection_name
from clinical_rag.errors import IngestError
from clinical_rag.eval.freeze import freeze_combo, load_selected
from clinical_rag.schemas import (
    ChromaConfig,
    EmbedConfig,
    EmbedProvider,
    PineconeConfig,
    VectorStoreKind,
    WeaviateConfig,
)


def test_openai_embed_fails_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("clinical_rag.adapters.embedders._load_dotenv_keys", lambda: None)
    with pytest.raises(IngestError, match="OPENAI_API_KEY"):
        build_embedder(EmbedConfig(provider=EmbedProvider.openai, model_id="text-embedding-3-small"))


def test_pinecone_store_fails_without_key():
    with pytest.raises(IngestError, match="PINECONE__API_KEY"):
        build_store(VectorStoreKind.pinecone, pinecone=PineconeConfig(api_key=""))


def test_weaviate_store_fails_without_url():
    with pytest.raises(IngestError, match="WEAVIATE__URL"):
        build_store(VectorStoreKind.weaviate, weaviate=WeaviateConfig(url=""))


def test_chroma_store_builds():
    store = build_store(VectorStoreKind.chroma, chroma=ChromaConfig(persist_dir="artifacts/indexes/chroma"))
    assert store is not None


def test_vector_store_env(monkeypatch):
    from clinical_rag.config import Settings

    monkeypatch.setenv("VECTOR_STORE", "weaviate")
    settings = Settings()
    assert settings.vector_store is VectorStoreKind.weaviate


def test_weaviate_collection_name_is_legal():
    name = weaviate_collection_name("demo__section_aware__BAAI_bge-m3")
    assert name[0].isupper()
    assert "-" not in name


def test_pinecone_index_names_from_sdk_shapes():
    class Listed:
        def names(self):
            return ["ragroog", "other"]

    assert pinecone_index_names(Listed()) == {"ragroog", "other"}
    assert pinecone_index_names([{"name": "ragroog"}]) == {"ragroog"}


def test_freeze_and_load(tmp_path: Path):
    path = tmp_path / "selected.yaml"
    combo = {
        "parser_engine": "pymupdf",
        "parser_profile": "ocr_fallback",
        "chunk": {"strategy_id": "section_aware", "target_tokens": 400, "overlap_ratio": 0.12},
        "embed": {"model_id": "BAAI/bge-small-en-v1.5"},
        "vector_store": "chroma",
        "retrieval": {"mode": "dense", "top_k": 5},
    }
    freeze_combo(combo, path)
    loaded = load_selected(path)
    assert loaded["parser_engine"] == "pymupdf"
    assert loaded["retrieval"]["mode"] == "dense"
