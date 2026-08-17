from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class MediaType(str, Enum):
    pdf = "pdf"
    md = "md"
    txt = "txt"
    json = "json"


class ExtractionMethod(str, Enum):
    text = "text"
    ocr = "ocr"
    hybrid = "hybrid"
    na = "n/a"


class StrategyId(str, Enum):
    fixed = "fixed"
    section_aware = "section_aware"
    hierarchical = "hierarchical"
    passthrough = "passthrough"


class ParserProfile(str, Enum):
    text_only = "text_only"
    ocr_fallback = "ocr_fallback"
    ocr_all = "ocr_all"


class LegalFlags(BaseModel):
    open_access_or_reusable: bool = False
    redistribution_ok_for_indexing: bool = False
    edition_current: bool = False
    attribution_documented: bool = False

    def complete(self) -> bool:
        return all(
            (
                self.open_access_or_reusable,
                self.redistribution_ok_for_indexing,
                self.edition_current,
                self.attribution_documented,
            )
        )


class RawDocument(BaseModel):
    doc_id: str
    filename: str
    media_type: MediaType
    document_name: str
    source_url: str = ""
    path: str
    legal: LegalFlags


class TextBlock(BaseModel):
    kind: str  # heading | paragraph | table
    text: str
    heading_level: int | None = None


class ParsedPage(BaseModel):
    page_number: int
    text: str
    extraction_method: ExtractionMethod
    blocks: list[TextBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    document_name: str
    source_url: str = ""
    media_type: MediaType
    filename: str = ""
    pages: list[ParsedPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    section_title: str
    page_number: int | None = None
    source_url: str = ""
    strategy_id: StrategyId
    corpus_id: str
    job_id: str
    extraction_method: ExtractionMethod
    token_count: int
    embed_model_id: str
    parent_chunk_id: str | None = None
    filename: str = ""
    doc_id: str = ""

    @model_validator(mode="after")
    def pdf_pages_required(self) -> Self:
        if self.extraction_method in (ExtractionMethod.text, ExtractionMethod.ocr, ExtractionMethod.hybrid):
            if self.filename.lower().endswith(".pdf") and self.page_number is None:
                raise ValueError("PDF-derived chunks must include page_number")
        return self


class SmokeQueryConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)


class ChunkConfig(BaseModel):
    strategy_id: StrategyId = StrategyId.section_aware
    target_tokens: int = 400
    overlap_ratio: float = Field(default=0.12, ge=0.0, lt=1.0)
    min_tokens: int = 120
    max_tokens: int = 520
    child_tokens: int = 350
    parent_tokens: int = 800


class EmbedConfig(BaseModel):
    model_id: str = "BAAI/bge-m3"
    fallback_model_id: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    batch_size: int = Field(default=16, ge=1)


class ChromaConfig(BaseModel):
    persist_dir: str = "artifacts/indexes/chroma"


class ParserConfig(BaseModel):
    profile: ParserProfile = ParserProfile.ocr_fallback
    ocr_lang: str = "eng"
    ocr_dpi: int = 250
    min_chars: int = 50
    min_alnum_ratio: float = 0.3


class IngestJobConfig(BaseModel):
    corpus_id: str
    job_id: str
    files: list[RawDocument]
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    smoke_query: SmokeQueryConfig = Field(default_factory=SmokeQueryConfig)
    rationale_chunk: str = ""
    rationale_embed: str = ""


class SmokeHit(BaseModel):
    score: float
    text: str
    document_name: str
    section_title: str
    page_number: int | None
    chunk_id: str
    extraction_method: str = ""
    source_url: str = ""
    token_count: int = 0


class IngestReport(BaseModel):
    job_id: str
    corpus_id: str
    collection_name: str
    strategy_id: str
    embed_model_id: str
    embed_device: str = "auto"
    parser_profile: str
    page_count: int
    ocr_page_count: int
    chunk_count: int
    warnings: list[str] = Field(default_factory=list)
    rationale: dict[str, str] = Field(default_factory=dict)
    rebuild_policy: str = (
        "Replace the target collection on each job so reruns are idempotent "
        "(same corpus/strategy/model name does not duplicate chunks)."
    )
