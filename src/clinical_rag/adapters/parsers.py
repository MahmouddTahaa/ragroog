from __future__ import annotations

import os
from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.parsing.base import ParseOutcome
from clinical_rag.parsing.cleanup import strip_repeating_headers_footers
from clinical_rag.parsing.pdf_pymupdf import parse_pdf
from clinical_rag.parsing.router import parse_document as parse_document_default
from clinical_rag.schemas import (
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    ParserConfig,
    ParserEngine,
    RawDocument,
    TextBlock,
)


def parse_with_engine(
    raw: RawDocument,
    parser: ParserConfig,
    cache_dir: Path | None,
) -> ParseOutcome:
    suffix = Path(raw.path).suffix.lower()
    # Non-PDF / non-engine-specific types always use the default router (md/txt/json/xml).
    if suffix != ".pdf" or parser.engine is ParserEngine.pymupdf:
        outcome = parse_document_default(raw, parser, cache_dir)
        if outcome.parsed is not None and suffix == ".pdf":
            outcome.parsed = strip_repeating_headers_footers(outcome.parsed)
        return outcome

    if parser.engine is ParserEngine.unstructured:
        parsed = _parse_unstructured(raw)
        parsed = strip_repeating_headers_footers(parsed)
        return ParseOutcome(parsed=parsed, warnings=list(parsed.warnings))

    if parser.engine is ParserEngine.llamaparse:
        parsed = _parse_llamaparse(raw)
        parsed = strip_repeating_headers_footers(parsed)
        return ParseOutcome(parsed=parsed, warnings=list(parsed.warnings))

    raise IngestError(f"Unknown parser engine: {parser.engine}")


def _parse_unstructured(raw: RawDocument) -> ParsedDocument:
    try:
        from unstructured.partition.auto import partition  # type: ignore
    except ImportError as exc:
        raise IngestError(
            "Parser engine 'unstructured' requires optional package. "
            "Install with: uv sync --extra unstructured"
        ) from exc
    elements = partition(filename=raw.path)
    if not elements:
        raise IngestError(f"{raw.filename}: unstructured returned no elements")
    blocks: list[TextBlock] = []
    texts: list[str] = []
    for el in elements:
        kind = "heading" if "Title" in type(el).__name__ or "Header" in type(el).__name__ else "paragraph"
        text = str(el).strip()
        if not text:
            continue
        texts.append(text)
        blocks.append(
            TextBlock(kind=kind, text=text, heading_level=1 if kind == "heading" else None)
        )
    page = ParsedPage(
        page_number=1,
        text="\n\n".join(texts),
        extraction_method=ExtractionMethod.text,
        blocks=blocks,
    )
    return ParsedDocument(
        doc_id=raw.doc_id,
        document_name=raw.document_name,
        source_url=raw.source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=[page],
        warnings=["unstructured: page_number approximated as 1"],
    )


def _parse_llamaparse(raw: RawDocument) -> ParsedDocument:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or os.environ.get("LLAMAPARSE_API_KEY")
    if not api_key:
        raise IngestError(
            "Parser engine 'llamaparse' requires LLAMA_CLOUD_API_KEY (API). Fail closed."
        )
    try:
        from llama_parse import LlamaParse  # type: ignore
    except ImportError as exc:
        raise IngestError(
            "Parser engine 'llamaparse' requires optional package. "
            "Install with: uv sync --extra llamaparse"
        ) from exc
    parser = LlamaParse(api_key=api_key, result_type="markdown")
    docs = parser.load_data(raw.path)
    if not docs:
        raise IngestError(f"{raw.filename}: LlamaParse returned no documents")
    pages: list[ParsedPage] = []
    for i, doc in enumerate(docs, start=1):
        text = getattr(doc, "text", None) or str(doc)
        pages.append(
            ParsedPage(
                page_number=i,
                text=text,
                extraction_method=ExtractionMethod.text,
                blocks=[TextBlock(kind="paragraph", text=text)] if text.strip() else [],
            )
        )
    return ParsedDocument(
        doc_id=raw.doc_id,
        document_name=raw.document_name,
        source_url=raw.source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=pages,
        warnings=["llamaparse: API parser"],
    )


# Keep pymupdf reference for type checkers / unused-import silence when routing PDFs.
_ = parse_pdf
