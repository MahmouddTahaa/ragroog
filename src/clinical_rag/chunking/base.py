from __future__ import annotations

from dataclasses import dataclass

from clinical_rag.schemas import (
    Chunk,
    ExtractionMethod,
    IngestJobConfig,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    StrategyId,
    TextBlock,
)

UNKNOWN = "(unknown)"
# ~4 characters per token. Offline stand-in for tiktoken; good enough to pack 300–500 token windows.
_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def encode_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [text[i : i + _CHARS_PER_TOKEN] for i in range(0, len(text), _CHARS_PER_TOKEN)]


def decode_tokens(tokens: list[str]) -> str:
    return "".join(tokens)


def window_ranges(n: int, size: int, overlap_ratio: float) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    size = max(1, size)
    step = max(1, int(size * (1 - overlap_ratio)))
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(n, i + size)
        out.append((i, j))
        if j == n:
            break
        i += step
    return out


def window_texts(text: str, size: int, overlap_ratio: float) -> list[str]:
    tokens = encode_tokens(text)
    if not tokens:
        return []
    if len(tokens) <= size:
        return [text]
    return [decode_tokens(tokens[i:j]) for i, j in window_ranges(len(tokens), size, overlap_ratio)]


@dataclass
class DraftChunk:
    text: str
    section_title: str
    page_number: int | None
    extraction_method: ExtractionMethod
    chunk_id: str | None = None
    parent_chunk_id: str | None = None


def fallback_blocks(page: ParsedPage) -> list[TextBlock]:
    if page.blocks:
        return page.blocks
    if page.text.strip():
        return [TextBlock(kind="paragraph", text=page.text)]
    return []


def first_heading(page: ParsedPage) -> str:
    for block in fallback_blocks(page):
        if block.kind == "heading" and block.text.strip():
            return block.text.strip()
    return UNKNOWN


def stamp_chunk(
    draft: DraftChunk,
    *,
    raw: RawDocument,
    config: IngestJobConfig,
    strategy_id: StrategyId,
    index: int,
) -> Chunk:
    chunk_id = draft.chunk_id or f"{raw.doc_id}-{strategy_id.value}-{index:04d}"
    return Chunk(
        chunk_id=chunk_id,
        text=draft.text,
        document_name=raw.document_name or draft.text[:80],
        section_title=draft.section_title or UNKNOWN,
        page_number=draft.page_number,
        source_url=raw.source_url,
        strategy_id=strategy_id,
        corpus_id=config.corpus_id,
        job_id=config.job_id,
        extraction_method=draft.extraction_method,
        token_count=count_tokens(draft.text),
        embed_model_id=config.embed.model_id,
        parent_chunk_id=draft.parent_chunk_id,
        filename=raw.filename,
        doc_id=raw.doc_id,
    )


def stamp_prechunked(chunk: Chunk, config: IngestJobConfig) -> Chunk:
    data = chunk.model_dump()
    data["corpus_id"] = config.corpus_id
    data["job_id"] = config.job_id
    data["embed_model_id"] = config.embed.model_id
    data["strategy_id"] = StrategyId.passthrough
    data["token_count"] = count_tokens(chunk.text)
    return Chunk.model_validate(data)


def token_stream(
    parsed: ParsedDocument,
) -> tuple[list[str], list[int], list[ExtractionMethod], list[str]]:
    tokens: list[str] = []
    pages: list[int] = []
    methods: list[ExtractionMethod] = []
    titles: list[str] = []
    for page in parsed.pages:
        piece = page.text if page.text.endswith("\n") else page.text + "\n"
        encoded = encode_tokens(piece)
        title = first_heading(page)
        tokens.extend(encoded)
        pages.extend([page.page_number] * len(encoded))
        methods.extend([page.extraction_method] * len(encoded))
        titles.extend([title] * len(encoded))
    return tokens, pages, methods, titles
