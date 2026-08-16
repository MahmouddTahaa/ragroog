from __future__ import annotations

from clinical_rag.indexing.chroma_store import ChromaStore
from clinical_rag.indexing.embed import Embedder
from clinical_rag.schemas import SmokeHit


def run_smoke_query(
    *,
    persist_dir: str,
    collection: str,
    embedder: Embedder,
    query: str,
    top_k: int,
) -> list[SmokeHit]:
    vectors = embedder.encode([query])
    store = ChromaStore(persist_dir)
    return store.query(collection, vectors[0], top_k)
