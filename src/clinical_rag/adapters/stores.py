from __future__ import annotations

import re
from typing import Any, Protocol
from urllib.parse import urlparse

from clinical_rag.errors import IngestError
from clinical_rag.indexing.chroma_store import ChromaStore
from clinical_rag.schemas import (
    ChromaConfig,
    Chunk,
    PineconeConfig,
    SmokeHit,
    VectorStoreKind,
    WeaviateConfig,
)


class VectorStore(Protocol):
    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def index_embed_model_id(self, name: str) -> str | None: ...

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]: ...


def build_store(
    kind: VectorStoreKind,
    *,
    chroma: ChromaConfig | None = None,
    weaviate: WeaviateConfig | None = None,
    pinecone: PineconeConfig | None = None,
) -> VectorStore:
    if kind is VectorStoreKind.chroma:
        return ChromaStore((chroma or ChromaConfig()).persist_dir)
    if kind is VectorStoreKind.weaviate:
        return WeaviateStore(weaviate or WeaviateConfig())
    if kind is VectorStoreKind.pinecone:
        return PineconeStore(pinecone or PineconeConfig())
    raise IngestError(f"Unknown vector store: {kind}")


def _hit_from_meta(meta: dict, *, text: str, score: float) -> SmokeHit:
    page = meta.get("page_number")
    return SmokeHit(
        score=round(float(score), 4),
        text=text or "",
        document_name=str(meta.get("document_name") or ""),
        section_title=str(meta.get("section_title") or ""),
        page_number=None if page in (None, -1, "-1") else int(page),
        chunk_id=str(meta.get("chunk_id") or ""),
        extraction_method=str(meta.get("extraction_method") or ""),
        source_url=str(meta.get("source_url") or ""),
        token_count=int(meta.get("token_count") or 0),
    )


def _flat_meta(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_name": chunk.document_name,
        "section_title": chunk.section_title,
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "source_url": chunk.source_url or "",
        "strategy_id": chunk.strategy_id.value,
        "corpus_id": chunk.corpus_id,
        "job_id": chunk.job_id,
        "extraction_method": chunk.extraction_method.value,
        "token_count": chunk.token_count,
        "embed_model_id": chunk.embed_model_id,
        "filename": chunk.filename,
        "doc_id": chunk.doc_id,
        "parent_chunk_id": chunk.parent_chunk_id or "",
        "text": chunk.text,
    }


def weaviate_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not cleaned:
        cleaned = "Collection"
    if cleaned[0].isdigit():
        cleaned = f"C_{cleaned}"
    return cleaned[0].upper() + cleaned[1:]


def _weaviate_is_cloud(host: str) -> bool:
    host = host.lower()
    return host.endswith("weaviate.cloud") or host.endswith("weaviate.io") or host.endswith(
        "weaviate.network"
    )


def _connect_weaviate(cfg: WeaviateConfig):
    import weaviate  # type: ignore
    from weaviate.auth import AuthApiKey  # type: ignore

    url = cfg.url.strip()
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    auth = AuthApiKey(cfg.api_key) if cfg.api_key.strip() else None
    try:
        if _weaviate_is_cloud(host):
            if not cfg.api_key.strip():
                raise IngestError("Weaviate Cloud requires WEAVIATE__API_KEY. Fail closed.")
            return weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=auth)
        http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        local = host in {"localhost", "127.0.0.1"} and parsed.scheme in {"http", "https", ""}
        if local:
            return weaviate.connect_to_local(
                host=host,
                port=http_port,
                grpc_port=cfg.grpc_port,
                auth_credentials=auth,
            )
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=parsed.scheme == "https",
            grpc_host=host,
            grpc_port=cfg.grpc_port,
            grpc_secure=parsed.scheme == "https",
            auth_credentials=auth,
        )
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Could not connect to Weaviate at {cfg.url}: {exc}") from exc


def _weaviate_vector_kwargs(Configure) -> dict:
    vectors = getattr(Configure, "Vectors", None)
    if vectors is not None and hasattr(vectors, "self_provided"):
        return {"vector_config": vectors.self_provided()}
    return {"vectorizer_config": Configure.Vectorizer.none()}


class WeaviateStore:
    """Deck option: open-source hybrid-capable store. Dense upsert/query for the lab."""

    def __init__(self, cfg: WeaviateConfig):
        self.cfg = cfg
        if not (cfg.url or "").strip():
            raise IngestError(
                "Vector store 'weaviate' requires WEAVIATE__URL "
                "(e.g. http://localhost:8080). Fail closed."
            )
        try:
            import weaviate  # noqa: F401  # type: ignore
        except ImportError as exc:
            raise IngestError(
                "Vector store 'weaviate' requires optional package. "
                "uv sync --extra weaviate"
            ) from exc
        self._client = _connect_weaviate(cfg)

    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from weaviate.classes.config import Configure, DataType, Property  # type: ignore
        from weaviate.classes.data import DataObject  # type: ignore

        col_name = weaviate_collection_name(name)
        if self._client.collections.exists(col_name):
            self._client.collections.delete(col_name)
        self._client.collections.create(
            name=col_name,
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="document_name", data_type=DataType.TEXT),
                Property(name="section_title", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="strategy_id", data_type=DataType.TEXT),
                Property(name="corpus_id", data_type=DataType.TEXT),
                Property(name="job_id", data_type=DataType.TEXT),
                Property(name="extraction_method", data_type=DataType.TEXT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="embed_model_id", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="parent_chunk_id", data_type=DataType.TEXT),
            ],
            **_weaviate_vector_kwargs(Configure),
        )
        collection = self._client.collections.get(col_name)
        objects = [
            DataObject(properties=_flat_meta(chunk), vector=vec)
            for chunk, vec in zip(chunks, embeddings, strict=True)
        ]
        result = collection.data.insert_many(objects)
        errors = getattr(result, "errors", None) or {}
        if getattr(result, "has_errors", False) or errors:
            raise IngestError(f"Weaviate insert_many failed: {errors}")

    def index_embed_model_id(self, name: str) -> str | None:
        col_name = weaviate_collection_name(name)
        if not self._client.collections.exists(col_name):
            return None
        collection = self._client.collections.get(col_name)
        for obj in collection.iterator(include_vector=False):
            props = obj.properties or {}
            model = props.get("embed_model_id")
            if model:
                return str(model)
            break
        return None

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]:
        from weaviate.classes.query import MetadataQuery  # type: ignore

        col_name = weaviate_collection_name(name)
        collection = self._client.collections.get(col_name)
        result = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        hits: list[SmokeHit] = []
        for obj in result.objects:
            props = dict(obj.properties or {})
            distance = getattr(obj.metadata, "distance", None)
            score = 1.0 - float(distance) if distance is not None else 0.0
            hits.append(_hit_from_meta(props, text=str(props.get("text") or ""), score=score))
        return hits


def pinecone_index_names(listed: Any) -> set[str]:
    names_fn = getattr(listed, "names", None)
    if callable(names_fn):
        return {str(n) for n in names_fn() if n}
    out: set[str] = set()
    for item in listed or []:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("index_name") or ""
            if name:
                out.add(str(name))
        else:
            name = getattr(item, "name", "") or ""
            if name:
                out.add(str(name))
    return out


def _pinecone_attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _pinecone_matches(res: Any) -> list:
    matches = _pinecone_attr(res, "matches", default=[])
    return list(matches or [])


def _pinecone_match_meta(match: Any) -> tuple[dict, float]:
    meta = _pinecone_attr(match, "metadata", default={}) or {}
    if not isinstance(meta, dict):
        meta = dict(meta)
    score = float(_pinecone_attr(match, "score", default=0.0) or 0.0)
    return dict(meta), score


class PineconeStore:
    """Deck option: managed hosted vector DB (API). Uses one index + namespace per collection."""

    def __init__(self, cfg: PineconeConfig):
        self.cfg = cfg
        if not (cfg.api_key or "").strip():
            raise IngestError(
                "Vector store 'pinecone' requires PINECONE__API_KEY (API). Fail closed."
            )
        if not (cfg.index_name or "").strip():
            raise IngestError("Vector store 'pinecone' requires PINECONE__INDEX_NAME.")
        try:
            from pinecone import Pinecone  # type: ignore
        except ImportError as exc:
            raise IngestError(
                "Vector store 'pinecone' requires optional package. "
                "uv sync --extra pinecone"
            ) from exc
        self._pc = Pinecone(api_key=cfg.api_key.strip())
        self._index_name = cfg.index_name.strip()

    def _ensure_index(self, dim: int) -> None:
        from pinecone import ServerlessSpec  # type: ignore

        existing = pinecone_index_names(self._pc.list_indexes())
        if self._index_name in existing:
            return
        self._pc.create_index(
            name=self._index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cfg.cloud, region=self.cfg.region),
        )

    def _index(self):
        return self._pc.Index(self._index_name)

    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not embeddings:
            raise IngestError("Pinecone upsert requires at least one embedding")
        self._ensure_index(len(embeddings[0]))
        index = self._index()
        try:
            index.delete(delete_all=True, namespace=name)
        except Exception:
            pass
        batch: list[dict] = []
        for chunk, vec in zip(chunks, embeddings, strict=True):
            meta = {k: v for k, v in _flat_meta(chunk).items() if v is not None}
            batch.append({"id": chunk.chunk_id, "values": vec, "metadata": meta})
            if len(batch) >= 100:
                index.upsert(vectors=batch, namespace=name)
                batch = []
        if batch:
            index.upsert(vectors=batch, namespace=name)

    def index_embed_model_id(self, name: str) -> str | None:
        index = self._index()
        stats = index.describe_index_stats()
        namespaces = _pinecone_attr(stats, "namespaces", default={}) or {}
        ns_stats = namespaces.get(name) if isinstance(namespaces, dict) else None
        count = int(_pinecone_attr(ns_stats, "vector_count", default=0) or 0)
        if count == 0:
            return None
        try:
            dim = int(_pinecone_attr(stats, "dimension", default=0) or 0)
            if not dim:
                return None
            res = index.query(
                vector=[0.0] * dim,
                top_k=1,
                namespace=name,
                include_metadata=True,
            )
            matches = _pinecone_matches(res)
            if not matches:
                return None
            meta, _score = _pinecone_match_meta(matches[0])
            model = meta.get("embed_model_id")
            return str(model) if model else None
        except Exception:
            return None

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]:
        index = self._index()
        res = index.query(
            vector=embedding,
            top_k=top_k,
            namespace=name,
            include_metadata=True,
        )
        hits: list[SmokeHit] = []
        for match in _pinecone_matches(res):
            meta, score = _pinecone_match_meta(match)
            hits.append(
                _hit_from_meta(
                    meta,
                    text=str(meta.get("text") or ""),
                    score=score,
                )
            )
        return hits


__all__ = [
    "ChromaStore",
    "Chunk",
    "PineconeStore",
    "SmokeHit",
    "VectorStore",
    "WeaviateStore",
    "build_store",
    "pinecone_index_names",
    "weaviate_collection_name",
]
