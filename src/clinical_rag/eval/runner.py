from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.errors import IngestError
from clinical_rag.eval.metrics import (
    aggregate_mean,
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from clinical_rag.config import get_settings, store_kwargs
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.schemas import ChromaConfig, EmbedProvider


@dataclass
class EvalQuestion:
    id: str
    query: str
    relevant_chunk_ids: list[str]
    notes: str = ""


@dataclass
class EvalRunResult:
    run_id: str
    metrics: dict
    per_query: list[dict] = field(default_factory=list)
    run_dir: Path | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    if not path.is_file():
        raise IngestError(f"Questions file not found: {path}")
    out: list[EvalQuestion] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}:{line_no}: invalid JSONL ({exc})") from exc
        qid = str(row.get("id") or f"q{line_no}")
        query = str(row.get("query") or "").strip()
        gold = row.get("relevant_chunk_ids") or []
        if not query:
            raise IngestError(f"{path}:{line_no}: missing query")
        if not isinstance(gold, list) or not gold:
            raise IngestError(f"{path}:{line_no}: relevant_chunk_ids must be a non-empty list")
        out.append(
            EvalQuestion(
                id=qid,
                query=query,
                relevant_chunk_ids=[str(x) for x in gold],
                notes=str(row.get("notes") or ""),
            )
        )
    if not out:
        raise IngestError(f"No questions in {path}")
    return out


def run_retrieval_eval(
    *,
    job_report: dict,
    questions: list[EvalQuestion],
    persist_dir: str,
    k_values: list[int],
    evals_dir: Path,
    eval_set_id: str = "default",
) -> EvalRunResult:
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not k_values:
        raise IngestError("k_values must include at least one positive integer")

    model_id = job_report["embed_model_id"]
    device = job_report.get("embed_device") or "auto"
    provider = job_report.get("embed_provider") or EmbedProvider.sentence_transformers.value
    collection = job_report["collection_name"]
    embedder = embedder_for_index(
        model_id=model_id,
        provider=provider,
        device=device,
        batch_size=8,
        purpose="query",
    )

    per_query: list[dict] = []
    latencies_ms: list[float] = []
    metric_bags: dict[str, list[float]] = {
        f"precision@{k}": [] for k in k_values
    }
    metric_bags.update({f"recall@{k}": [] for k in k_values})
    metric_bags.update({f"hit@{k}": [] for k in k_values})
    metric_bags.update({f"ndcg@{k}": [] for k in k_values})
    mrr_values: list[float] = []

    settings = get_settings()
    chroma_cfg = ChromaConfig(persist_dir=persist_dir) if persist_dir else settings.chroma
    stores = store_kwargs(settings)
    stores["chroma"] = chroma_cfg
    max_k = max(k_values)
    for q in questions:
        t0 = time.perf_counter()
        hits = run_smoke_query(
            persist_dir=chroma_cfg.persist_dir,
            collection=collection,
            embedder=embedder,
            query=q.query,
            top_k=max_k,
            index_model_id=model_id,
            vector_store=job_report.get("vector_store", "chroma"),
            **stores,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        retrieved = [h.chunk_id for h in hits]
        relevant = set(q.relevant_chunk_ids)
        row = {
            "id": q.id,
            "query": q.query,
            "relevant_chunk_ids": q.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved,
            "latency_ms": round(elapsed_ms, 2),
            "scores": {},
        }
        for k in k_values:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            h = hit_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            metric_bags[f"precision@{k}"].append(p)
            metric_bags[f"recall@{k}"].append(r)
            metric_bags[f"hit@{k}"].append(h)
            metric_bags[f"ndcg@{k}"].append(n)
            row["scores"][f"precision@{k}"] = p
            row["scores"][f"recall@{k}"] = r
            row["scores"][f"hit@{k}"] = h
            row["scores"][f"ndcg@{k}"] = n
        mrr = reciprocal_rank(retrieved, relevant)
        mrr_values.append(mrr)
        row["scores"]["mrr"] = mrr
        per_query.append(row)

    run_id = uuid.uuid4().hex[:12]
    combo = job_report.get("combo") or {
        "parser_engine": job_report.get("parser_engine"),
        "parser_profile": job_report.get("parser_profile"),
        "chunk": {"strategy_id": job_report.get("strategy_id")},
        "embed": {"model_id": model_id},
        "vector_store": job_report.get("vector_store", "chroma"),
        "retrieval": {"mode": job_report.get("retrieval_mode", "dense")},
    }
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_report.get("job_id"),
        "collection_name": collection,
        "eval_set_id": eval_set_id,
        "k_values": k_values,
        "n_questions": len(questions),
        "combo": combo,
        "aggregates": {
            **{name: round(aggregate_mean(vals), 4) for name, vals in metric_bags.items()},
            "mrr": round(aggregate_mean(mrr_values), 4),
            "latency_ms_p50": round(percentile(latencies_ms, 50), 2),
            "latency_ms_p95": round(percentile(latencies_ms, 95), 2),
        },
    }

    evals_dir = Path(evals_dir)
    run_dir = evals_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_query:
            fh.write(json.dumps(row) + "\n")
    _append_leaderboard(evals_dir / "leaderboard.csv", metrics)
    return EvalRunResult(run_id=run_id, metrics=metrics, per_query=per_query, run_dir=run_dir)


def _append_leaderboard(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combo = metrics.get("combo") or {}
    chunk = combo.get("chunk") or {}
    embed = combo.get("embed") or {}
    retrieval = combo.get("retrieval") or {}
    aggregates = metrics.get("aggregates") or {}
    row = {
        "run_id": metrics.get("run_id"),
        "timestamp": metrics.get("timestamp"),
        "job_id": metrics.get("job_id"),
        "collection_name": metrics.get("collection_name"),
        "eval_set_id": metrics.get("eval_set_id"),
        "parser_engine": combo.get("parser_engine"),
        "parser_profile": combo.get("parser_profile"),
        "chunk_strategy": chunk.get("strategy_id"),
        "embed_model_id": embed.get("model_id"),
        "vector_store": combo.get("vector_store"),
        "retrieval_mode": retrieval.get("mode"),
        "precision@3": aggregates.get("precision@3"),
        "precision@5": aggregates.get("precision@5"),
        "precision@10": aggregates.get("precision@10"),
        "recall@5": aggregates.get("recall@5"),
        "hit@5": aggregates.get("hit@5"),
        "mrr": aggregates.get("mrr"),
        "ndcg@5": aggregates.get("ndcg@5"),
        "latency_ms_p50": aggregates.get("latency_ms_p50"),
        "latency_ms_p95": aggregates.get("latency_ms_p95"),
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_leaderboard(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
