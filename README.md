# ragroog

Offline **corpus factory** for a clinical RAG hackathon (Day 1): upload → legal gate → parse (PyMuPDF + Tesseract fallback) → chunk → local embeddings → Chroma → smoke query (top-n).

No LLM generation, hybrid search, or rerank in this slice.

## Setup

Python 3.11–3.13 (the repo pins 3.12). [uv](https://docs.astral.sh/uv/) is the intended installer.

```bash
uv sync --extra dev
```

Optional OCR (scanned PDFs):

```bash
# Fedora
sudo dnf install tesseract tesseract-langpack-eng
```

## Run

```bash
uv run streamlit run src/clinical_rag/ui/streamlit_app.py
```

CLI twin:

```bash
uv run python scripts/build_index.py --confirm-legal \
  --corpus-id demo \
  --smoke-query "first-line hypertension treatment" \
  data/guidelines/sample_hypertension.md
```

## Tests

```bash
uv run pytest
```

Tests use a hash embedder so they do not download `BAAI/bge-m3`. The UI/CLI default still uses bge-m3 locally.

## Samples

`data/guidelines/` holds synthetic demo text (not clinical advice): markdown, raw JSON, and pre-chunked JSON.
