from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from clinical_rag.chunking import chunk_parsed
from clinical_rag.chunking.base import stamp_prechunked
from clinical_rag.config import (
    DEFAULT_CHUNK_RATIONALE,
    DEFAULT_EMBED_RATIONALE,
    collection_name,
)
from clinical_rag.errors import IngestError
from clinical_rag.indexing.chroma_store import ChromaStore
from clinical_rag.indexing.embed import Embedder, SentenceTransformerEmbedder
from clinical_rag.parsing.router import parse_document
from clinical_rag.schemas import (
    Chunk,
    ExtractionMethod,
    IngestJobConfig,
    IngestReport,
    StrategyId,
)

ProgressFn = Callable[[str, str], None]


def _progress(fn: ProgressFn | None, stage: str, message: str) -> None:
    if fn:
        fn(stage, message)


def run_ingest(
    config: IngestJobConfig,
    *,
    progress: ProgressFn | None = None,
    embedder: Embedder | None = None,
    jobs_dir: Path | None = None,
) -> tuple[IngestReport, list[Chunk]]:
    incomplete = [d.filename for d in config.files if not d.legal.complete()]
    if incomplete:
        raise IngestError(f"Legal checklist incomplete for: {', '.join(incomplete)}")
    if not config.files:
        raise IngestError("No files to ingest")

    jobs_dir = Path(jobs_dir or "artifacts/jobs")
    job_dir = jobs_dir / config.job_id
    cache_dir = job_dir / "parsed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    warnings: list[str] = []
    page_count = 0
    ocr_page_count = 0

    for raw in config.files:
        _progress(progress, "parse", f"Parsing {raw.filename}")
        outcome = parse_document(raw, config.parser, cache_dir)
        warnings.extend(outcome.warnings)
        if outcome.prechunked is not None:
            if config.chunk.strategy_id is not StrategyId.passthrough:
                warnings.append(
                    f"{raw.filename}: pre-chunked JSON used passthrough (not re-chunked)"
                )
            chunks.extend(stamp_prechunked(c, config) for c in outcome.prechunked)
            continue
        if outcome.parsed is None:
            raise IngestError(f"{raw.filename}: parser returned nothing")
        parsed = outcome.parsed
        warnings.extend(parsed.warnings)
        page_count += len(parsed.pages)
        ocr_page_count += sum(
            1
            for p in parsed.pages
            if p.extraction_method in (ExtractionMethod.ocr, ExtractionMethod.hybrid)
        )
        if config.chunk.strategy_id is StrategyId.passthrough:
            raise IngestError("passthrough is only valid for pre-chunked JSON")
        chunks.extend(chunk_parsed(parsed, raw, config))

    if not chunks:
        raise IngestError("No chunks produced")

    _progress(progress, "embed", f"Embedding {len(chunks)} chunks ({config.embed.model_id})")
    embedder = embedder or SentenceTransformerEmbedder(config.embed)
    vectors = embedder.encode([c.text for c in chunks])
    for chunk in chunks:
        chunk.embed_model_id = embedder.model_id

    name = collection_name(config.corpus_id, config.chunk.strategy_id.value, embedder.model_id)
    _progress(progress, "store", f"Writing collection {name}")
    store = ChromaStore(config.chroma.persist_dir)
    store.replace(name, chunks, vectors)

    report = IngestReport(
        job_id=config.job_id,
        corpus_id=config.corpus_id,
        collection_name=name,
        strategy_id=config.chunk.strategy_id.value,
        embed_model_id=embedder.model_id,
        parser_profile=config.parser.profile.value,
        page_count=page_count,
        ocr_page_count=ocr_page_count,
        chunk_count=len(chunks),
        warnings=warnings,
        rationale={
            "chunk": config.rationale_chunk or DEFAULT_CHUNK_RATIONALE,
            "embed": config.rationale_embed or DEFAULT_EMBED_RATIONALE,
        },
    )
    (job_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    payload = [c.model_dump(mode="json") for c in chunks]
    (job_dir / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report, chunks
