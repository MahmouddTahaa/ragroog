from __future__ import annotations

from pathlib import Path

import yaml

from clinical_rag.errors import IngestError


def freeze_combo(combo: dict, path: Path | str = "configs/selected.yaml") -> Path:
    """Write the winning lab combination for a future product UI to load."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    required = ("parser_engine", "chunk", "embed", "vector_store")
    missing = [k for k in required if k not in combo]
    if missing:
        raise IngestError(f"Cannot freeze combo; missing keys: {', '.join(missing)}")
    payload = {
        "parser_engine": combo.get("parser_engine"),
        "parser_profile": combo.get("parser_profile", "ocr_fallback"),
        "chunk": combo.get("chunk") or {},
        "embed": combo.get("embed") or {},
        "vector_store": combo.get("vector_store", "chroma"),
        "retrieval": combo.get("retrieval") or {"mode": "dense", "top_k": 5},
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def load_selected(path: Path | str = "configs/selected.yaml") -> dict:
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"No frozen config at {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
