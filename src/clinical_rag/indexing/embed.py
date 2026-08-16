from __future__ import annotations

from typing import Protocol

from clinical_rag.config import detect_device
from clinical_rag.errors import IngestError
from clinical_rag.schemas import EmbedConfig


class Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbedConfig):
        self.model_id = cfg.model_id
        self._fallback = cfg.fallback_model_id
        self._device = detect_device(cfg.device)
        self._batch_size = cfg.batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(self.model_id, device=self._device)
        except Exception as exc:
            if not self._fallback or self._fallback == self.model_id:
                raise IngestError(f"Failed to load embed model {self.model_id}: {exc}") from exc
            try:
                self._model = SentenceTransformer(self._fallback, device=self._device)
                self.model_id = self._fallback
            except Exception as exc2:
                raise IngestError(
                    f"Failed to load {self.model_id} and fallback {self._fallback}: {exc2}"
                ) from exc2

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
