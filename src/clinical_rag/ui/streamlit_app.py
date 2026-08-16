from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from clinical_rag.config import (
    DEFAULT_CHUNK_RATIONALE,
    DEFAULT_EMBED_RATIONALE,
    collection_name,
    get_settings,
    make_doc_id,
    make_job_id,
)
from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import SentenceTransformerEmbedder
from clinical_rag.parsing.router import media_type_for
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
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

st.set_page_config(page_title="Ragroog ingest", layout="wide")
settings = get_settings()
JOBS = Path(settings.jobs_dir)
UPLOADS = Path(settings.uploads_dir)


def _jobs() -> list[Path]:
    if not JOBS.exists():
        return []
    return sorted((p for p in JOBS.iterdir() if (p / "report.json").exists()), reverse=True)


def _load_report(job_dir: Path) -> dict:
    return json.loads((job_dir / "report.json").read_text(encoding="utf-8"))


@st.cache_resource
def _embedder(model_id: str, device: str, batch_size: int) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        EmbedConfig(model_id=model_id, device=device, batch_size=batch_size)
    )


st.title("Ragroog — Day 1 offline ingest")
st.caption("Upload → legal gate → parse/OCR → chunk → embed → Chroma → smoke top-n. No generation.")

ingest_tab, browse_tab, smoke_tab, report_tab = st.tabs(
    ["Ingest", "Browse chunks", "Smoke query", "Job report"]
)

with ingest_tab:
    files = st.file_uploader(
        "Guidelines (pdf, md, txt, json)",
        accept_multiple_files=True,
        type=["pdf", "md", "txt", "json"],
    )
    corpus_id = st.text_input("corpus_id", value="demo")
    c1, c2, c3, c4 = st.columns(4)
    legal = LegalFlags(
        open_access_or_reusable=c1.checkbox("Open access / reusable"),
        redistribution_ok_for_indexing=c2.checkbox("OK to index"),
        edition_current=c3.checkbox("Edition current"),
        attribution_documented=c4.checkbox("Attribution documented"),
    )
    if files:
        st.subheader("Per-file metadata")
        names: dict[str, str] = {}
        urls: dict[str, str] = {}
        for uploaded in files:
            cols = st.columns((2, 3))
            names[uploaded.name] = cols[0].text_input(
                "document_name",
                value=Path(uploaded.name).stem.replace("_", " "),
                key=f"name-{uploaded.name}",
            )
            urls[uploaded.name] = cols[1].text_input(
                "source_url", value="", key=f"url-{uploaded.name}"
            )
    else:
        names, urls = {}, {}

    st.subheader("Operator knobs")
    k1, k2, k3 = st.columns(3)
    profile = k1.selectbox(
        "Parser profile",
        options=[p.value for p in ParserProfile],
        index=list(ParserProfile).index(settings.parser.profile),
    )
    strategy = k2.selectbox(
        "Chunk strategy",
        options=[s.value for s in StrategyId],
        index=list(StrategyId).index(settings.chunk.strategy_id),
    )
    embed_model = k3.selectbox(
        "Embed model",
        options=["BAAI/bge-m3", "BAAI/bge-small-en-v1.5"],
        index=0 if settings.embed.model_id == "BAAI/bge-m3" else 1,
    )
    p1, p2, p3 = st.columns(3)
    target_tokens = p1.number_input("target_tokens", 64, 2000, settings.chunk.target_tokens)
    overlap = p2.number_input(
        "overlap_ratio", 0.0, 0.5, float(settings.chunk.overlap_ratio), 0.01
    )
    device = p3.selectbox("device", ["auto", "cpu", "cuda", "mps"], index=0)

    rationale_chunk = st.text_area("Chunk rationale", value=DEFAULT_CHUNK_RATIONALE, height=80)
    rationale_embed = st.text_area("Embed rationale", value=DEFAULT_EMBED_RATIONALE, height=80)

    if st.button("Build index", type="primary", disabled=not files):
        if not legal.complete():
            st.error("Legal checklist must be complete before ingest.")
        else:
            job_id = make_job_id()
            dest = UPLOADS / job_id
            dest.mkdir(parents=True, exist_ok=True)
            raw_docs: list[RawDocument] = []
            used_ids: dict[str, int] = {}
            for uploaded in files:
                path = dest / uploaded.name
                path.write_bytes(uploaded.getvalue())
                doc_id = make_doc_id(uploaded.name)
                used_ids[doc_id] = used_ids.get(doc_id, 0) + 1
                if used_ids[doc_id] > 1:
                    doc_id = f"{doc_id}-{used_ids[doc_id]}"
                raw_docs.append(
                    RawDocument(
                        doc_id=doc_id,
                        filename=uploaded.name,
                        media_type=media_type_for(uploaded.name),
                        document_name=names.get(uploaded.name) or Path(uploaded.name).stem,
                        source_url=urls.get(uploaded.name) or "",
                        path=str(path),
                        legal=legal,
                    )
                )
            config = IngestJobConfig(
                corpus_id=corpus_id.strip() or "demo",
                job_id=job_id,
                files=raw_docs,
                parser=ParserConfig(profile=ParserProfile(profile)),
                chunk=ChunkConfig(
                    strategy_id=StrategyId(strategy),
                    target_tokens=int(target_tokens),
                    overlap_ratio=float(overlap),
                ),
                embed=EmbedConfig(model_id=embed_model, device=device),
                chroma=settings.chroma,
                smoke_query=settings.smoke_query,
                rationale_chunk=rationale_chunk,
                rationale_embed=rationale_embed,
            )
            status = st.status("Running ingest…", expanded=True)
            try:

                def on_progress(stage: str, message: str) -> None:
                    status.write(f"**{stage}:** {message}")

                report, chunks = run_ingest(config, progress=on_progress, jobs_dir=JOBS)
                status.update(label="Index built", state="complete")
            except IngestError as exc:
                status.update(label="Ingest failed", state="error")
                st.error(str(exc))
            else:
                st.session_state["last_job_id"] = report.job_id
                st.session_state["last_collection"] = report.collection_name
                st.session_state["last_embed_model"] = report.embed_model_id
                st.success(
                    f"{report.chunk_count} chunks · {report.page_count} pages · "
                    f"{report.ocr_page_count} OCR pages · `{report.collection_name}`"
                )
                ocr_notes = [w for w in report.warnings if "ocr" in w.lower() or "tesseract" in w.lower()]
                for note in ocr_notes:
                    st.warning(note)
                if report.warnings and not ocr_notes:
                    st.info("\n".join(report.warnings[:8]))

with browse_tab:
    job_dirs = _jobs()
    labels = [p.name for p in job_dirs]
    default = st.session_state.get("last_job_id")
    idx = labels.index(default) if default in labels else 0
    selected = st.selectbox("Job", labels, index=idx if labels else 0) if labels else None
    if selected:
        chunks_path = JOBS / selected / "chunks.json"
        rows = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []
        docs = sorted({r.get("document_name") or "" for r in rows})
        strategies = sorted({r.get("strategy_id") or "" for r in rows})
        methods = sorted({r.get("extraction_method") or "" for r in rows})
        f1, f2, f3 = st.columns(3)
        doc_f = f1.selectbox("document", ["(all)", *docs])
        strat_f = f2.selectbox("strategy", ["(all)", *strategies])
        meth_f = f3.selectbox("extraction_method", ["(all)", *methods])
        filtered = [
            r
            for r in rows
            if (doc_f == "(all)" or r.get("document_name") == doc_f)
            and (strat_f == "(all)" or r.get("strategy_id") == strat_f)
            and (meth_f == "(all)" or r.get("extraction_method") == meth_f)
        ]
        st.caption(f"{len(filtered)} chunks")
        for row in filtered:
            title = f"{row.get('chunk_id')} · {row.get('section_title')} · p{row.get('page_number')}"
            with st.expander(title):
                st.json({k: v for k, v in row.items() if k != "text"})
                st.text(row.get("text") or "")

with smoke_tab:
    job_dirs = _jobs()
    labels = [p.name for p in job_dirs]
    default = st.session_state.get("last_job_id")
    idx = labels.index(default) if default in labels else 0
    selected = st.selectbox("Job", labels, index=idx if labels else 0, key="smoke-job") if labels else None
    query = st.text_input("Query")
    top_k = st.number_input("top_k / n", 1, 50, settings.smoke_query.top_k)
    if selected and st.button("Run smoke query") and query.strip():
        report = _load_report(JOBS / selected)
        try:
            embedder = _embedder(report["embed_model_id"], settings.embed.device, settings.embed.batch_size)
            hits = run_smoke_query(
                persist_dir=settings.chroma.persist_dir,
                collection=report["collection_name"],
                embedder=embedder,
                query=query.strip(),
                top_k=int(top_k),
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["last_collection"] = report["collection_name"]
            table = [
                {
                    "score": h.score,
                    "document_name": h.document_name,
                    "section_title": h.section_title,
                    "page_number": h.page_number,
                    "chunk_id": h.chunk_id,
                    "excerpt": (h.text[:240] + "…") if len(h.text) > 240 else h.text,
                }
                for h in hits
            ]
            st.dataframe(table, use_container_width=True)
            for h in hits:
                with st.expander(f"{h.score:.3f} · {h.chunk_id}"):
                    st.write(h.text)

with report_tab:
    job_dirs = _jobs()
    labels = [p.name for p in job_dirs]
    default = st.session_state.get("last_job_id")
    idx = labels.index(default) if default in labels else 0
    selected = st.selectbox("Job", labels, index=idx if labels else 0, key="report-job") if labels else None
    if selected:
        path = JOBS / selected / "report.json"
        report = _load_report(JOBS / selected)
        st.json(report)
        st.download_button("Download report.json", path.read_bytes(), file_name="report.json")
        chunk_r = st.text_area(
            "Chunk rationale",
            value=report.get("rationale", {}).get("chunk") or DEFAULT_CHUNK_RATIONALE,
            key="edit-chunk-r",
        )
        embed_r = st.text_area(
            "Embed rationale",
            value=report.get("rationale", {}).get("embed") or DEFAULT_EMBED_RATIONALE,
            key="edit-embed-r",
        )
        if st.button("Save rationale"):
            report.setdefault("rationale", {})
            report["rationale"]["chunk"] = chunk_r
            report["rationale"]["embed"] = embed_r
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            st.success("Saved")
        st.caption(
            collection_name(
                report.get("corpus_id") or "demo",
                report.get("strategy_id") or "section_aware",
                report.get("embed_model_id") or "BAAI/bge-m3",
            )
        )
