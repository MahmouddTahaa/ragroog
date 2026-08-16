from __future__ import annotations

from pathlib import Path

from clinical_rag.schemas import Chunk, SmokeHit


def _meta(chunk: Chunk) -> dict:
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
    }


class ChromaStore:
    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _client_obj(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        client = self._client_obj()
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_meta(c) for c in chunks],
        )

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]:
        col = self._client_obj().get_collection(name)
        count = col.count()
        n = min(top_k, max(count, 1))
        if count == 0:
            return []
        result = col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SmokeHit] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            meta = meta or {}
            page = meta.get("page_number")
            hits.append(
                SmokeHit(
                    score=round(1.0 - float(dist), 4),
                    text=text or "",
                    document_name=str(meta.get("document_name") or ""),
                    section_title=str(meta.get("section_title") or ""),
                    page_number=None if page in (None, -1) else int(page),
                    chunk_id=str(meta.get("chunk_id") or ""),
                    extraction_method=str(meta.get("extraction_method") or ""),
                    source_url=str(meta.get("source_url") or ""),
                    token_count=int(meta.get("token_count") or 0),
                )
            )
        return hits

    def list_chunks(self, name: str, limit: int = 500) -> list[dict]:
        col = self._client_obj().get_collection(name)
        count = min(col.count(), limit)
        if count == 0:
            return []
        raw = col.get(limit=count, include=["documents", "metadatas"])
        rows = []
        for cid, text, meta in zip(raw.get("ids") or [], raw.get("documents") or [], raw.get("metadatas") or []):
            row = dict(meta or {})
            row["chunk_id"] = cid
            row["text"] = text
            rows.append(row)
        return rows
