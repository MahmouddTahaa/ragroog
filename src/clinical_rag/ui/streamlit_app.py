from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.config import (
    get_settings,
    make_doc_id,
    make_job_id,
    resolve_embed_config,
    store_kwargs,
)
from clinical_rag.errors import IngestError
from clinical_rag.eval.freeze import freeze_combo
from clinical_rag.eval.runner import load_leaderboard, load_questions, run_retrieval_eval
from clinical_rag.parsing.router import media_type_for
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    EmbedProvider,
    IngestJobConfig,
    LegalFlags,
    ParserConfig,
    ParserEngine,
    ParserProfile,
    PineconeConfig,
    RawDocument,
    RetrievalConfig,
    RetrievalMode,
    StrategyId,
    VectorStoreKind,
    WeaviateConfig,
)

CITATION_FIELDS = ("document_name", "section_title", "page_number", "chunk_id")
PAGE_SIZE = 25
EVALS_DIR = Path("artifacts/evals")
SELECTED_PATH = Path("configs/selected.yaml")

LEGAL_HELP = {
    "open_access_or_reusable": "Source is open access or you have reuse rights for this project.",
    "redistribution_ok_for_indexing": "Creating a searchable vector index from this content is allowed.",
    "edition_current": "You are indexing the current edition, not a superseded version.",
    "attribution_documented": "Provenance is recorded (document_name / source_url).",
}

st.set_page_config(
    page_title="Ragroog lab",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "Ragroog testing & tuning lab (not the product UI).",
    },
)
settings = get_settings()
JOBS = Path(settings.jobs_dir)
UPLOADS = Path(settings.uploads_dir)


def _jobs() -> list[Path]:
    if not JOBS.exists():
        return []
    return sorted((p for p in JOBS.iterdir() if (p / "report.json").exists()), reverse=True)


def _load_report(job_dir: Path) -> dict:
    return json.loads((job_dir / "report.json").read_text(encoding="utf-8"))


def _job_label(job_dir: Path) -> str:
    report = _load_report(job_dir)
    engine = report.get("parser_engine") or "pymupdf"
    store = report.get("vector_store") or "chroma"
    embed = report.get("embed_model_id") or "?"
    if "/" in embed:
        embed = embed.rsplit("/", 1)[-1]
    return (
        f"{report.get('corpus_id', '?')} · {report.get('chunk_count', '?')} · "
        f"{report.get('strategy_id', '?')} · {embed} · {engine}/{store} · {job_dir.name[:8]}"
    )


def _job_options() -> tuple[list[str], dict[str, Path]]:
    dirs = _jobs()
    mapping = {_job_label(p): p for p in dirs}
    return list(mapping.keys()), mapping


@st.cache_resource
def _query_embedder(model_id: str, provider: str, device: str, batch_size: int):
    return embedder_for_index(
        model_id=model_id,
        provider=provider,
        device=device,
        batch_size=batch_size,
        purpose="query",
    )


def _embed_models_for_provider(provider: str, *, include: str | None = None) -> list[str]:
    options = {
        EmbedProvider.sentence_transformers.value: [
            "BAAI/bge-m3",
            "BAAI/bge-small-en-v1.5",
        ],
        EmbedProvider.openai.value: [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
        EmbedProvider.cohere.value: [
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
        ],
    }.get(provider, [])
    if include and include not in options:
        options = [include, *options]
    return options


def _show_citation_chunk(row: dict, *, score: float | None = None) -> None:
    title_bits = [str(row.get("chunk_id") or "")]
    if score is not None:
        title_bits.insert(0, f"{score:.3f}")
    title_bits.append(str(row.get("section_title") or ""))
    title_bits.append(f"p{row.get('page_number')}")
    with st.expander(" · ".join(title_bits)):
        for key in CITATION_FIELDS:
            st.write(f"**{key}:** {row.get(key)}")
        if score is not None:
            st.write(f"**score:** {score}")
        st.text(row.get("text") or "")


st.title("Ragroog — testing & tuning lab")
st.caption(
    "Operator lab for comparing parser × chunk × embed × store × retrieval combos. "
    "Not the patient/clinician product UI — a future product build will load "
    "`configs/selected.yaml` only."
)

ingest_tab, browse_tab, smoke_tab, report_tab, compare_tab = st.tabs(
    ["Ingest", "Browse chunks", "Smoke query", "Job report", "Compare"]
)

with ingest_tab:
    files = st.file_uploader(
        "Guidelines (pdf, md, txt, json, xml, nxml) — multi-file OK under one corpus_id",
        accept_multiple_files=True,
        type=["pdf", "md", "txt", "json", "xml", "nxml"],
    )
    corpus_id = st.text_input("corpus_id", value="demo", help="Operator namespace; not a disease enum.")

    st.subheader("Legal usability checklist")
    st.warning(
        "Ragroog does not verify licenses and is not responsible for misuse. "
        "You must confirm legal usability before indexing."
    )
    c1, c2 = st.columns(2)
    legal = LegalFlags(
        open_access_or_reusable=c1.checkbox(
            "Open access / reusable", help=LEGAL_HELP["open_access_or_reusable"]
        ),
        redistribution_ok_for_indexing=c2.checkbox(
            "OK to index (redistribution permitted)",
            help=LEGAL_HELP["redistribution_ok_for_indexing"],
        ),
        edition_current=c1.checkbox("Edition current", help=LEGAL_HELP["edition_current"]),
        attribution_documented=c2.checkbox(
            "Attribution documented", help=LEGAL_HELP["attribution_documented"]
        ),
    )

    names: dict[str, str] = {}
    urls: dict[str, str] = {}
    if files:
        st.subheader("Per-file metadata")
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

    st.subheader("Toolkit knobs (lab)")
    st.caption(
        "Deck Module 4: embed providers = OpenAI text-embedding-3 · Cohere embed-v3 · "
        "sentence-transformers (local). Vector DBs = Chroma · Pinecone · Weaviate."
    )
    resolved_embed, embed_hints = resolve_embed_config(settings.embed)
    t1, t2, t3, t4 = st.columns(4)
    parser_engine = t1.selectbox(
        "Parser engine",
        options=[e.value for e in ParserEngine],
        index=0,
        help="Default: pymupdf (local). unstructured / llamaparse need extras; llamaparse is API.",
    )
    profile = t2.selectbox(
        "Parser profile (pymupdf OCR)",
        options=[p.value for p in ParserProfile],
        index=list(ParserProfile).index(settings.parser.profile),
        disabled=parser_engine != ParserEngine.pymupdf.value,
    )
    strategy = t3.selectbox(
        "Chunk strategy",
        options=[s.value for s in StrategyId],
        index=list(StrategyId).index(settings.chunk.strategy_id),
    )
    vector_store = t4.selectbox(
        "Vector store",
        options=[v.value for v in VectorStoreKind],
        index=list(VectorStoreKind).index(settings.vector_store),
        help="Chroma = local default. Weaviate / Pinecone need URL or API key (configure below or in .env).",
    )

    weaviate_cfg = settings.weaviate
    pinecone_cfg = settings.pinecone
    if vector_store == VectorStoreKind.chroma.value:
        st.caption(f"Chroma persist dir: `{settings.chroma.persist_dir}`")
    elif vector_store == VectorStoreKind.weaviate.value:
        w1, w2, w3 = st.columns(3)
        weaviate_url = w1.text_input("Weaviate URL", value=settings.weaviate.url)
        weaviate_key = w2.text_input(
            "Weaviate API key (optional)",
            value=settings.weaviate.api_key,
            type="password",
        )
        weaviate_grpc = w3.number_input(
            "Weaviate gRPC port", 1, 65535, int(settings.weaviate.grpc_port)
        )
        weaviate_cfg = WeaviateConfig(
            url=weaviate_url.strip(),
            api_key=weaviate_key,
            grpc_port=int(weaviate_grpc),
        )
        st.caption(
            "Install: `uv sync --extra weaviate`. Local: `docker compose up -d` "
            "(http://localhost:8080). Cloud: cluster URL + API key."
        )
    elif vector_store == VectorStoreKind.pinecone.value:
        p1, p2, p3, p4 = st.columns(4)
        pine_key = p1.text_input(
            "Pinecone API key",
            value=settings.pinecone.api_key,
            type="password",
        )
        pine_index = p2.text_input("Pinecone index", value=settings.pinecone.index_name)
        pine_cloud = p3.text_input("Pinecone cloud", value=settings.pinecone.cloud)
        pine_region = p4.text_input("Pinecone region", value=settings.pinecone.region)
        pinecone_cfg = PineconeConfig(
            api_key=pine_key,
            index_name=pine_index.strip(),
            cloud=pine_cloud.strip() or "aws",
            region=pine_region.strip() or "us-east-1",
        )
        st.caption("Install: `uv sync --extra pinecone`. One serverless index; each job uses a namespace.")

    e1, e2, e3 = st.columns(3)
    embed_provider = e1.selectbox(
        "Embed provider (deck)",
        options=[
            EmbedProvider.sentence_transformers.value,
            EmbedProvider.openai.value,
            EmbedProvider.cohere.value,
        ],
        format_func=lambda v: {
            EmbedProvider.sentence_transformers.value: "sentence-transformers (local)",
            EmbedProvider.openai.value: "OpenAI text-embedding-3 (API)",
            EmbedProvider.cohere.value: "Cohere embed-v3 (API)",
        }.get(v, v),
        help="Matches Day 1 deck Module 4 model selection.",
    )
    if embed_provider == EmbedProvider.sentence_transformers.value:
        embed_options = ["BAAI/bge-m3", "BAAI/bge-small-en-v1.5"]
        embed_model = e2.selectbox(
            "Local ST model",
            options=embed_options,
            index=0 if resolved_embed.model_id == "BAAI/bge-m3" else 1,
            help="Concrete local sentence-transformers models (deck leaves the pick to you).",
        )
    elif embed_provider == EmbedProvider.openai.value:
        embed_model = e2.selectbox(
            "OpenAI model (API)",
            options=["text-embedding-3-small", "text-embedding-3-large"],
            help="Requires OPENAI_API_KEY in .env.",
        )
    else:
        embed_model = e2.selectbox(
            "Cohere model (API)",
            options=["embed-english-v3.0", "embed-multilingual-v3.0"],
            help="Requires COHERE_API_KEY in .env.",
        )
    device = e3.selectbox(
        "device (local ST only)",
        ["auto", "cpu", "cuda", "mps"],
        index=0,
        disabled=embed_provider != EmbedProvider.sentence_transformers.value,
    )
    if embed_hints and embed_provider == EmbedProvider.sentence_transformers.value:
        st.caption("Auto embed: " + "; ".join(embed_hints))

    strategy_id = StrategyId(strategy)
    target_tokens = settings.chunk.target_tokens
    overlap = float(settings.chunk.overlap_ratio)
    max_tokens = settings.chunk.max_tokens
    parent_tokens = settings.chunk.parent_tokens
    child_tokens = settings.chunk.child_tokens

    if strategy_id is StrategyId.fixed:
        p1, p2 = st.columns(2)
        target_tokens = p1.number_input("target_tokens", 64, 2000, settings.chunk.target_tokens)
        overlap = p2.number_input("overlap_ratio", 0.0, 0.5, float(settings.chunk.overlap_ratio), 0.01)
    elif strategy_id is StrategyId.section_aware:
        p1, p2, p3 = st.columns(3)
        target_tokens = p1.number_input("target_tokens", 64, 2000, settings.chunk.target_tokens)
        overlap = p2.number_input("overlap_ratio", 0.0, 0.5, float(settings.chunk.overlap_ratio), 0.01)
        max_tokens = p3.number_input("max_tokens", 64, 4000, settings.chunk.max_tokens)
    elif strategy_id is StrategyId.hierarchical:
        p1, p2, p3 = st.columns(3)
        parent_tokens = p1.number_input("parent_tokens", 64, 4000, settings.chunk.parent_tokens)
        child_tokens = p2.number_input("child_tokens", 32, 2000, settings.chunk.child_tokens)
        overlap = p3.number_input("overlap_ratio", 0.0, 0.5, float(settings.chunk.overlap_ratio), 0.01)

    retrieval_top_k = st.number_input(
        "retrieval top_k (dense baseline)", 1, 50, settings.smoke_query.top_k
    )

    if st.button("Build index", type="primary", disabled=not files):
        if not legal.complete():
            st.error("Legal usability checklist incomplete — all four flags required.")
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
                parser=ParserConfig(
                    engine=ParserEngine(parser_engine),
                    profile=ParserProfile(profile),
                ),
                chunk=ChunkConfig(
                    strategy_id=strategy_id,
                    target_tokens=int(target_tokens),
                    overlap_ratio=float(overlap),
                    max_tokens=int(max_tokens),
                    parent_tokens=int(parent_tokens),
                    child_tokens=int(child_tokens),
                ),
                embed=EmbedConfig(
                    model_id=embed_model,
                    device=device,
                    provider=EmbedProvider(embed_provider),
                ),
                chroma=settings.chroma,
                weaviate=weaviate_cfg,
                pinecone=pinecone_cfg,
                vector_store=VectorStoreKind(vector_store),
                retrieval=RetrievalConfig(mode=RetrievalMode.dense, top_k=int(retrieval_top_k)),
                smoke_query=settings.smoke_query,
            )
            status = st.status("Running ingest…", expanded=True)
            try:

                def on_progress(stage: str, message: str) -> None:
                    status.write(f"**{stage}:** {message}")

                report, _chunks = run_ingest(config, progress=on_progress, jobs_dir=JOBS)
                status.update(label="Index built", state="complete")
            except IngestError as exc:
                status.update(label="Ingest failed", state="error")
                st.error(str(exc))
            else:
                st.session_state["last_job_id"] = report.job_id
                st.session_state["last_collection"] = report.collection_name
                st.success(
                    f"{report.chunk_count} chunks · {report.page_count} pages · "
                    f"{report.ocr_page_count} OCR · `{report.collection_name}`"
                )
                st.json(report.combo)
                for note in report.warnings[:8]:
                    st.warning(note)

with browse_tab:
    labels, mapping = _job_options()
    default = st.session_state.get("last_job_id")
    default_label = next((lab for lab, p in mapping.items() if p.name == default), None)
    idx = labels.index(default_label) if default_label in labels else 0
    selected_label = st.selectbox("Job", labels, index=idx if labels else 0) if labels else None
    if selected_label:
        selected = mapping[selected_label]
        rows = json.loads((selected / "chunks.json").read_text(encoding="utf-8"))
        docs = sorted({r.get("document_name") or "" for r in rows})
        f1, f2 = st.columns(2)
        doc_f = f1.selectbox("document", ["(all)", *docs])
        text_q = f2.text_input("Search chunk_id / section_title / text", value="")
        filtered = []
        needle = text_q.strip().lower()
        for r in rows:
            if doc_f != "(all)" and r.get("document_name") != doc_f:
                continue
            if needle:
                blob = " ".join(
                    [
                        str(r.get("chunk_id") or ""),
                        str(r.get("section_title") or ""),
                        str(r.get("text") or ""),
                    ]
                ).lower()
                if needle not in blob:
                    continue
            filtered.append(r)
        total = len(filtered)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = st.number_input("Page", 1, pages, 1)
        start = (int(page) - 1) * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        st.caption(f"{start + 1 if total else 0}–{end} of {total}")
        for row in filtered[start:end]:
            _show_citation_chunk(row)

with smoke_tab:
    labels, mapping = _job_options()
    default = st.session_state.get("last_job_id")
    default_label = next((lab for lab, p in mapping.items() if p.name == default), None)
    idx = labels.index(default_label) if default_label in labels else 0
    selected_label = (
        st.selectbox("Job", labels, index=idx if labels else 0, key="smoke-job") if labels else None
    )
    query = st.text_input("Query", value="functional dyspepsia first line treatment")
    top_k = st.number_input("top_k / n", 1, 50, settings.smoke_query.top_k)

    query_provider = EmbedProvider.sentence_transformers.value
    query_model = settings.embed.model_id
    index_model = ""
    index_provider = EmbedProvider.sentence_transformers.value
    index_device = settings.embed.device

    if selected_label:
        report_preview = _load_report(mapping[selected_label])
        index_model = report_preview["embed_model_id"]
        index_provider = (
            report_preview.get("embed_provider") or EmbedProvider.sentence_transformers.value
        )
        index_device = report_preview.get("embed_device") or settings.embed.device
        st.caption(f"Index embedder for this job: `{index_provider}` / `{index_model}`")

        job_key = mapping[selected_label].name
        q1, q2 = st.columns(2)
        provider_opts = [
            EmbedProvider.sentence_transformers.value,
            EmbedProvider.openai.value,
            EmbedProvider.cohere.value,
        ]
        query_provider = q1.selectbox(
            "Query embed provider",
            options=provider_opts,
            index=provider_opts.index(index_provider)
            if index_provider in provider_opts
            else 0,
            key=f"smoke-query-provider-{job_key}",
            help="Must match the provider/model used when this job was indexed.",
        )
        model_opts = _embed_models_for_provider(
            query_provider,
            include=index_model if query_provider == index_provider else None,
        )
        model_default = (
            index_model
            if query_provider == index_provider and index_model in model_opts
            else model_opts[0]
        )
        query_model = q2.selectbox(
            "Query embed model",
            options=model_opts,
            index=model_opts.index(model_default),
            key=f"smoke-query-model-{job_key}-{query_provider}",
        )
        if query_provider != index_provider or query_model != index_model:
            st.warning(
                f"Selected query embedder differs from the index "
                f"({index_provider} / {index_model}). Retrieval will fail until they match."
            )

    if selected_label and st.button("Run smoke query") and query.strip():
        report = _load_report(mapping[selected_label])
        with st.spinner("Embedding query and retrieving…"):
            try:
                embedder = _query_embedder(
                    query_model,
                    query_provider,
                    index_device,
                    settings.embed.batch_size,
                )
                st.caption(f"Query embed: {embedder.model_id} ({query_provider}) on {embedder.device}")
                hits = run_smoke_query(
                    persist_dir=settings.chroma.persist_dir,
                    collection=report["collection_name"],
                    embedder=embedder,
                    query=query.strip(),
                    top_k=int(top_k),
                    index_model_id=index_model or report["embed_model_id"],
                    vector_store=report.get("vector_store", "chroma"),
                    **store_kwargs(settings),
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
                    }
                    for h in hits
                ]
                st.dataframe(table, use_container_width=True)
                for h in hits:
                    _show_citation_chunk(h.model_dump(), score=h.score)

with report_tab:
    labels, mapping = _job_options()
    default = st.session_state.get("last_job_id")
    default_label = next((lab for lab, p in mapping.items() if p.name == default), None)
    idx = labels.index(default_label) if default_label in labels else 0
    selected_label = (
        st.selectbox("Job", labels, index=idx if labels else 0, key="report-job") if labels else None
    )
    if selected_label:
        path = mapping[selected_label] / "report.json"
        report = _load_report(mapping[selected_label])
        st.subheader("Chosen combo")
        st.json(report.get("combo") or {})
        st.download_button("Download report.json", path.read_bytes(), file_name="report.json")
        with st.expander("Full report"):
            st.json(report)
        if st.button("Freeze this combo → configs/selected.yaml"):
            try:
                out = freeze_combo(report.get("combo") or {}, SELECTED_PATH)
                st.success(f"Wrote {out}")
            except IngestError as exc:
                st.error(str(exc))

with compare_tab:
    st.caption(
        "Day 2 measured baseline: Precision@K / Recall@K / Hit@K / MRR / latency per combination."
    )
    labels, mapping = _job_options()
    default = st.session_state.get("last_job_id")
    default_label = next((lab for lab, p in mapping.items() if p.name == default), None)
    idx = labels.index(default_label) if default_label in labels else 0
    selected_label = (
        st.selectbox("Job / collection", labels, index=idx if labels else 0, key="eval-job")
        if labels
        else None
    )
    q_path = st.text_input("Questions JSONL", value="data/eval/questions.jsonl")
    k_text = st.text_input("k list", value="3,5,10")
    col_a, col_b = st.columns(2)
    run_btn = col_a.button("Run eval", type="primary", disabled=not selected_label)
    reload_btn = col_b.button("Reload leaderboard")

    if run_btn and selected_label:
        report = _load_report(mapping[selected_label])
        with st.spinner("Running retrieval eval…"):
            try:
                k_values = [int(x.strip()) for x in k_text.split(",") if x.strip()]
                questions = load_questions(Path(q_path))
                result = run_retrieval_eval(
                    job_report=report,
                    questions=questions,
                    persist_dir=settings.chroma.persist_dir,
                    k_values=k_values,
                    evals_dir=EVALS_DIR,
                    eval_set_id=Path(q_path).stem,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Eval run {result.run_id}")
                st.json(result.metrics.get("aggregates") or {})
                with st.expander("Per-query breakdown"):
                    st.dataframe(result.per_query, use_container_width=True)

    rows = load_leaderboard(EVALS_DIR / "leaderboard.csv")
    if rows:
        st.subheader("Leaderboard")
        st.dataframe(rows, use_container_width=True)
        run_ids = [r.get("run_id") for r in rows if r.get("run_id")]
        pick = st.selectbox("Freeze winner from run", run_ids)
        if pick and st.button("Freeze this combo", key="freeze-eval"):
            metrics_path = EVALS_DIR / pick / "metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                out = freeze_combo(metrics.get("combo") or {}, SELECTED_PATH)
                st.success(f"Wrote {out}")
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("No leaderboard rows yet. Run eval on at least one job.")
