from __future__ import annotations

from clinical_rag.errors import IngestError
from clinical_rag.indexing.chroma_store import ChromaStore
from clinical_rag.indexing.embed import Embedder, assert_query_embedder_matches_index
from clinical_rag.schemas import SmokeHit


def run_smoke_query(
    *,
    persist_dir: str,
    collection: str,
    embedder: Embedder,
    query: str,
    top_k: int,
    index_model_id: str | None = None,
) -> list[SmokeHit]:
    store = ChromaStore(persist_dir)
    resolved_model = index_model_id or store.index_embed_model_id(collection)
    if not resolved_model:
        raise IngestError(f"Collection {collection!r} has no embed_model_id metadata")
    assert_query_embedder_matches_index(embedder, resolved_model)
    vectors = embedder.encode([query])
    return store.query(collection, vectors[0], top_k)
