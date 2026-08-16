"""CLI twin of the Streamlit ingest tab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.config import get_settings, make_doc_id, make_job_id
from clinical_rag.errors import IngestError
from clinical_rag.parsing.router import media_type_for
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.indexing.embed import SentenceTransformerEmbedder
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    IngestJobConfig,
    LegalFlags,
    ParserConfig,
    ParserProfile,
    RawDocument,
    StrategyId,
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Chroma index from local files")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--corpus-id", default="demo")
    p.add_argument("--strategy", default="section_aware", choices=[s.value for s in StrategyId])
    p.add_argument("--parser-profile", default="ocr_fallback", choices=[p.value for p in ParserProfile])
    p.add_argument("--embed-model", default="BAAI/bge-m3")
    p.add_argument("--source-url", default="")
    p.add_argument("--confirm-legal", action="store_true", help="Attest all four legal flags")
    p.add_argument("--smoke-query", default="")
    p.add_argument("--top-k", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _args()
    if not args.confirm_legal:
        raise SystemExit("Refusing to ingest without --confirm-legal")
    settings = get_settings()
    legal = LegalFlags(
        open_access_or_reusable=True,
        redistribution_ok_for_indexing=True,
        edition_current=True,
        attribution_documented=True,
    )
    files: list[RawDocument] = []
    used: dict[str, int] = {}
    for path in args.files:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        doc_id = make_doc_id(path.name)
        used[doc_id] = used.get(doc_id, 0) + 1
        if used[doc_id] > 1:
            doc_id = f"{doc_id}-{used[doc_id]}"
        files.append(
            RawDocument(
                doc_id=doc_id,
                filename=path.name,
                media_type=media_type_for(path.name),
                document_name=path.stem.replace("_", " "),
                source_url=args.source_url,
                path=str(path.resolve()),
                legal=legal,
            )
        )
    config = IngestJobConfig(
        corpus_id=args.corpus_id,
        job_id=make_job_id(),
        files=files,
        parser=ParserConfig(profile=ParserProfile(args.parser_profile)),
        chunk=ChunkConfig(strategy_id=StrategyId(args.strategy)),
        embed=EmbedConfig(model_id=args.embed_model, device=settings.embed.device),
        chroma=settings.chroma,
        smoke_query=settings.smoke_query,
    )
    try:
        report, _chunks = run_ingest(
            config,
            progress=lambda stage, msg: print(f"[{stage}] {msg}"),
            jobs_dir=settings.jobs_dir,
        )
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    print(report.model_dump_json(indent=2))
    query = args.smoke_query.strip()
    if query:
        embedder = SentenceTransformerEmbedder(config.embed)
        hits = run_smoke_query(
            persist_dir=config.chroma.persist_dir,
            collection=report.collection_name,
            embedder=embedder,
            query=query,
            top_k=args.top_k or settings.smoke_query.top_k,
        )
        for hit in hits:
            print(f"{hit.score:.3f}\t{hit.chunk_id}\t{hit.document_name}\t{hit.section_title}\tp{hit.page_number}")


if __name__ == "__main__":
    main()
