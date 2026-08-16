from pathlib import Path

from clinical_rag.chunking import chunk_parsed
from clinical_rag.parsing.text_plain import parse_plain
from clinical_rag.schemas import ChunkConfig, IngestJobConfig, StrategyId
from tests.helpers import raw_doc


def _config(tmp_path: Path, strategy: StrategyId, **chunk_kw) -> IngestJobConfig:
    md = tmp_path / "guide.md"
    md.write_text(
        "# Intro\n\nShort intro paragraph.\n\n"
        "## Treatment\n\nGive drug A then reassess. " * 40
        + "\n\n## Follow-up\n\nSee the patient again in two weeks.\n",
        encoding="utf-8",
    )
    raw = raw_doc(md, doc_id="guide")
    parsed = parse_plain(raw)
    cfg = IngestJobConfig(
        corpus_id="t",
        job_id="job1",
        files=[raw],
        chunk=ChunkConfig(strategy_id=strategy, **chunk_kw),
    )
    return cfg, raw, parsed


def test_section_aware_keeps_headings(tmp_path: Path):
    cfg, raw, parsed = _config(tmp_path, StrategyId.section_aware)
    chunks = chunk_parsed(parsed, raw, cfg)
    titles = {c.section_title for c in chunks}
    assert "Treatment" in titles or "Intro" in titles
    assert all(c.chunk_id.startswith("guide-section_aware-") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_fixed_windows_have_stable_ids(tmp_path: Path):
    cfg, raw, parsed = _config(tmp_path, StrategyId.fixed, target_tokens=80, overlap_ratio=0.12)
    a = chunk_parsed(parsed, raw, cfg)
    b = chunk_parsed(parsed, raw, cfg)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.token_count > 0 for c in a)


def test_hierarchical_indexes_children(tmp_path: Path):
    cfg, raw, parsed = _config(
        tmp_path,
        StrategyId.hierarchical,
        parent_tokens=120,
        child_tokens=50,
        overlap_ratio=0.12,
    )
    chunks = chunk_parsed(parsed, raw, cfg)
    assert chunks
    assert all(c.parent_chunk_id for c in chunks)
    assert all(c.parent_chunk_id.startswith("guide-parent-") for c in chunks)
